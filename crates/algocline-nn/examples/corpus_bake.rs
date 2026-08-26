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
//! Each `NN_BAKE_CORPUS` entry is a JSON file shaped like:
//!
//! ```json
//! { "meta": { "ctx_len": 48, "vocab_size": 46 }, "rows": [[3, 7, 1], [2, 9]] }
//! ```
//!
//! `rows` are already-tokenized id sequences (no tokenizer is loaded
//! here). Rows shorter than `ctx_len` are padded with `NN_BAKE_PAD_ID`,
//! longer rows are truncated. Any other `meta` field is ignored, so a
//! producer may record whatever else it wants alongside these two —
//! except the ones `meta.requires` names, for which see below.
//!
//! # Inputs (environment)
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `NN_BAKE_CORPUS` | *required* | Comma-separated corpus JSON paths |
//! | `NN_BAKE_OUT` | *required* | Output `*.safetensors` path |
//! | `NN_BAKE_STEPS` | *required* | Optimizer steps |
//! | `NN_BAKE_COND` | unset | Comma-separated condition slot per corpus, same arity as `NN_BAKE_CORPUS` |
//! | `NN_BAKE_COND_SLOTS` | `max(NN_BAKE_COND) + 1` | Conditioning table size |
//! | `NN_BAKE_MASK_LOGITS` | `0` | Score each target among the ids its position allowed |
//! | `NN_BAKE_ALLOWED_INPUT` | `0` | Hand the allowed ids to the model as an input channel |
//! | `NN_BAKE_BATCH` | `32` | Rows per step |
//! | `NN_BAKE_LR` | `3e-3` | Learning rate (constant schedule, no warmup) |
//! | `NN_BAKE_LAYERS` | `2` | Transformer blocks |
//! | `NN_BAKE_HEADS` | `4` | Attention heads (must divide `NN_BAKE_DIM`) |
//! | `NN_BAKE_DIM` | `128` | Hidden size |
//! | `NN_BAKE_PAD_ID` | `0` | Padding token id |
//!
//! Context length and vocabulary size are not settable: they come from
//! the corpus `meta`, and every corpus of one run must agree on them.
//!
//! The learning-rate schedule is fixed to constant with no warmup —
//! this driver serves one registered experiment series and that is a
//! condition of it, not a knob.
//!
//! # Conditioning
//!
//! Giving `NN_BAKE_COND` switches the run to the conditioned trainer:
//! the model grows a conditioning table of `NN_BAKE_COND_SLOTS` rows and
//! every corpus row is trained under the slot its own corpus was
//! labelled with. Without it, the plain trainer runs and the model
//! carries no table.
//!
//! Rows from several corpora are interleaved round-robin rather than
//! concatenated. Concatenation makes the source a function of how far
//! the run has got, which a conditioned run cannot separate from the
//! condition it is supposed to be binding. When the corpora are of
//! unequal size, the surplus of the largest ones trails at the end.
//!
//! # Allowed-id sets
//!
//! A corpus may state which ids each position of each row was allowed to
//! take. The format is opt-in and announces itself:
//!
//! ```json
//! {
//!   "meta": { "ctx_len": 4, "vocab_size": 10, "requires": ["per_row_allowed"] },
//!   "rows": [[3, 7, 1, 5]],
//!   "allowed": [{ "2": [7, 8], "4": [5] }]
//! }
//! ```
//!
//! `allowed` is parallel to `rows`, one entry per row, and each entry is
//! sparse: it maps a **1-based** position to the ids available there.
//! A position nobody listed is unconstrained, which is also what an
//! empty set means downstream — so the padding past the end of a row is
//! left out rather than spelled.
//!
//! `meta.requires` is the format's forward-compatibility list, and this
//! driver implements exactly one entry, `per_row_allowed`. A corpus
//! requiring anything else is refused by name instead of being read as
//! though the field it names were decoration. The list and the field
//! are checked against each other in both directions: `allowed` without
//! the requirement would let an older reader train on the same rows
//! unconstrained and report the same numbers, and the requirement
//! without `allowed` is a producer that meant to write sets and did not.
//!
//! Two independent switches consume the sets, and at least one has to,
//! or the run is refused — sets that are loaded and then unused make a
//! plain run wearing the label of a constrained one:
//!
//! - `NN_BAKE_MASK_LOGITS` scores each target among the ids its position
//!   allowed rather than among the whole vocabulary.
//! - `NN_BAKE_ALLOWED_INPUT` gives the model the set as an input, so it
//!   is told what is available before it answers.
//!
//! `NN_BAKE_ALLOWED_INPUT` cannot be combined with `NN_BAKE_COND`: the
//! architecture refuses a model carrying both tables, because neither
//! forward pass delivers both channels and the one such a model would go
//! through would drop the channel the caller paid for.
//!
//! # Outputs
//!
//! - `NN_BAKE_OUT` — the final weights.
//! - stdout — one JSON line:
//!   `{"steps":…,"final_loss":…,"min_loss":…,"elapsed_s":…,"out":…,"rows":…,"corpus_rows":…,"cond_slots":…,"allowed_positions":…,"mask_disallowed_logits":…,"allowed_input":…,"device":…,"ctx_len":…,"vocab_size":…}`
//!   (`cond_slots` is `null` for an unconditioned run, and
//!   `allowed_positions` is `null` for a run with no allowed-id sets).
//! - stderr — progress lines, plus the per-step loss when the caller
//!   sets `RUST_LOG=algocline_nn=info`.
//!
//! Every malformed or inconsistent input is refused by name before any
//! training starts; nothing here falls back to a default on bad input.
//!
//! # Usage
//!
//! ```bash
//! # CUDA (requires nvcc + the nn-cuda feature at compile time)
//! NN_BAKE_CORPUS=/data/a.json,/data/b.json \
//! NN_BAKE_COND=0,1 \
//! NN_BAKE_STEPS=250 \
//! NN_BAKE_OUT=/data/out.safetensors \
//!   cargo run --release --features nn-cuda --example corpus_bake
//!
//! # CPU (small shapes only)
//! NN_BAKE_CORPUS=/data/a.json NN_BAKE_STEPS=2 NN_BAKE_OUT=/tmp/o.safetensors \
//!   cargo run --release --example corpus_bake
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use algocline_nn::arch::{CondIndex, Gpt2Config, Gpt2Custom, Gpt2Model};
use algocline_nn::train::{
    run_allowed_ft, run_conditioned_ft, run_full_ft, CrossEntropyLoss, DatasetOpts, FullFtConfig,
    ScheduleKind, TokenizedDataset, TrainingLease,
};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde::Deserialize;

/// One corpus file as it sits on disk. Unknown `meta` fields are
/// ignored on purpose — a producer records more than a trainer reads —
/// with the exception of the ones `meta.requires` names.
#[derive(Debug, Deserialize)]
struct CorpusFile {
    meta: CorpusMeta,
    rows: Vec<Vec<u32>>,
    /// The allowed ids per row, sparse by 1-based position, parallel to
    /// [`Self::rows`]. Present exactly when `meta.requires` lists
    /// `per_row_allowed`; the two are checked against each other.
    #[serde(default)]
    allowed: Option<Vec<BTreeMap<String, Vec<u32>>>>,
}

/// The `meta` fields this driver takes its shape from, plus the
/// forward-compatibility list.
#[derive(Debug, Deserialize)]
struct CorpusMeta {
    ctx_len: usize,
    vocab_size: usize,
    /// Held as raw JSON rather than a typed list so a value of the wrong
    /// shape is refused here, naming the field, rather than surfacing as
    /// a parse error about the corpus as a whole.
    #[serde(default)]
    requires: Option<serde_json::Value>,
}

/// A corpus after loading: the shape it declares, its rows, and, when it
/// carries them, its allowed-id sets in the dense per-position form the
/// dataset takes.
#[derive(Debug)]
struct LoadedCorpus {
    ctx_len: usize,
    vocab_size: usize,
    rows: Vec<Vec<u32>>,
    allowed: Option<Vec<Vec<Vec<u32>>>>,
}

/// A loaded corpus together with the condition slot it was labelled
/// with, if the run is conditioned.
#[derive(Debug)]
struct Source {
    path: PathBuf,
    rows: Vec<Vec<u32>>,
    /// Dense allowed-id sets, `[row][position]`, in this source's own row
    /// order — the pairing the merge below has to preserve.
    allowed: Option<Vec<Vec<Vec<u32>>>>,
    cond: Option<u32>,
}

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

/// Parse an optional `usize`. A value that is present but unparseable
/// is refused rather than replaced by `default` — a typo'd step count
/// that silently becomes the default is a run nobody can interpret.
fn env_usize(key: &str, default: usize) -> Result<usize, String> {
    match env_str(key)? {
        None => Ok(default),
        Some(v) => v
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("{key}={v:?} is not a non-negative integer ({e})")),
    }
}

fn env_required_usize(key: &str, what: &str) -> Result<usize, String> {
    let raw = env_required(key, what)?;
    raw.trim()
        .parse::<usize>()
        .map_err(|e| format!("{key}={raw:?} is not a non-negative integer ({e})"))
}

fn env_u32(key: &str, default: u32) -> Result<u32, String> {
    match env_str(key)? {
        None => Ok(default),
        Some(v) => v
            .trim()
            .parse::<u32>()
            .map_err(|e| format!("{key}={v:?} is not a non-negative integer ({e})")),
    }
}

fn env_f64(key: &str, default: f64) -> Result<f64, String> {
    match env_str(key)? {
        None => Ok(default),
        Some(v) => {
            let parsed = v
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("{key}={v:?} is not a number ({e})"))?;
            if !parsed.is_finite() {
                return Err(format!("{key}={v:?} is not a finite number"));
            }
            Ok(parsed)
        }
    }
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

/// Parse `NN_BAKE_COND` into one slot id per corpus path.
///
/// The two lists are paired positionally, so a length disagreement
/// means some corpus would be trained under another corpus's condition
/// with every shape still agreeing.
fn parse_conds(raw: &str, corpus_count: usize) -> Result<Vec<u32>, String> {
    let parts = split_list("NN_BAKE_COND", raw)?;
    if parts.len() != corpus_count {
        return Err(format!(
            "NN_BAKE_COND names {} condition(s) for the {corpus_count} corpus path(s) in \
             NN_BAKE_CORPUS — the pairing is positional, so it takes exactly one per corpus",
            parts.len()
        ));
    }
    let mut slots = Vec::with_capacity(parts.len());
    for part in &parts {
        slots.push(
            part.parse::<u32>()
                .map_err(|e| format!("NN_BAKE_COND entry {part:?} is not a slot id ({e})"))?,
        );
    }
    Ok(slots)
}

// ─── Corpus loading ─────────────────────────────────────────────

/// The optional format extensions this driver implements, and so the
/// entries it accepts in `meta.requires`.
const UNDERSTOOD_REQUIREMENTS: [&str; 1] = ["per_row_allowed"];

/// Short kind label for a JSON value. The value itself is deliberately
/// not printed: a corpus row is long, and what a reader needs is which
/// part disagreed rather than its content.
fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Read `meta.requires`, returning whether it asks for the per-row
/// allowed-id sets.
///
/// Absent or empty is the older format — every unknown `meta` field is
/// ignored. An entry outside [`UNDERSTOOD_REQUIREMENTS`] is refused by
/// name rather than ignored: the producer listed it because the rows do
/// not mean what they say without it.
fn requires_per_row_allowed(
    path: &Path,
    requires: Option<&serde_json::Value>,
) -> Result<bool, String> {
    let Some(value) = requires else {
        return Ok(false);
    };
    let listed = value.as_array().ok_or_else(|| {
        format!(
            "NN_BAKE_CORPUS entry {path:?} declares meta.requires as a JSON {}, not an \
             array of requirement names",
            json_kind(value)
        )
    })?;
    let mut wanted = false;
    for (index, entry) in listed.iter().enumerate() {
        let name = entry.as_str().ok_or_else(|| {
            format!(
                "NN_BAKE_CORPUS entry {path:?} declares meta.requires[{index}] as a JSON {}, \
                 not a requirement name",
                json_kind(entry)
            )
        })?;
        if !UNDERSTOOD_REQUIREMENTS.contains(&name) {
            return Err(format!(
                "NN_BAKE_CORPUS entry {path:?} requires {name:?}, which this driver does not \
                 implement (it implements {UNDERSTOOD_REQUIREMENTS:?}) — the producer listed \
                 it because the rows do not mean what they say without it"
            ));
        }
        wanted = true;
    }
    Ok(wanted)
}

/// Turn one corpus's sparse allowed-id maps into the dense
/// `[row][position]` form [`TokenizedDataset::with_allowed_ids`] takes.
///
/// The keys are 1-based positions and the maps are sparse, so a row's
/// dense list runs to its own last listed position and holds the empty
/// set — "unconstrained" — everywhere nobody listed. Rows are widened
/// to a common length later, once every source has been read, because
/// the model's allowed-id input refuses rows describing differing
/// numbers of positions.
fn densify_allowed(
    path: &Path,
    sparse: &[BTreeMap<String, Vec<u32>>],
    rows: &[Vec<u32>],
    vocab: u32,
) -> Result<Vec<Vec<Vec<u32>>>, String> {
    if sparse.len() != rows.len() {
        return Err(format!(
            "NN_BAKE_CORPUS entry {path:?} carries {} allowed entr(ies) for {} row(s) — the \
             two are parallel, so it takes exactly one entry per row",
            sparse.len(),
            rows.len()
        ));
    }
    let mut dense = Vec::with_capacity(sparse.len());
    for (row_idx, map) in sparse.iter().enumerate() {
        let mut listed: Vec<(usize, &Vec<u32>)> = Vec::with_capacity(map.len());
        for (key, ids) in map {
            let position = key.trim().parse::<usize>().map_err(|e| {
                format!(
                    "NN_BAKE_CORPUS entry {path:?} allowed[{row_idx}] is keyed by {key:?}, \
                     which is not a 1-based position ({e})"
                )
            })?;
            if position == 0 {
                return Err(format!(
                    "NN_BAKE_CORPUS entry {path:?} allowed[{row_idx}] holds a set at position \
                     0 — the positions are 1-based"
                ));
            }
            if let Some(id) = ids.iter().copied().find(|id| *id >= vocab) {
                return Err(format!(
                    "NN_BAKE_CORPUS entry {path:?} allowed[{row_idx}] position {position} \
                     allows token id {id}, outside the meta.vocab_size {vocab} it declares"
                ));
            }
            listed.push((position, ids));
        }
        let width = listed.iter().map(|(p, _)| *p).max().unwrap_or(0);
        let mut row = vec![Vec::new(); width];
        for (position, ids) in listed {
            row[position - 1] = ids.clone();
        }
        dense.push(row);
    }
    Ok(dense)
}

/// Read and validate one corpus file.
fn load_corpus(path: &Path) -> Result<LoadedCorpus, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("NN_BAKE_CORPUS entry {path:?} could not be read: {e}"))?;
    let corpus: CorpusFile = serde_json::from_str(&text).map_err(|e| {
        format!(
            "NN_BAKE_CORPUS entry {path:?} is not a corpus JSON object \
             ({{\"meta\":{{\"ctx_len\":N,\"vocab_size\":V}},\"rows\":[[id,…]]}}): {e}"
        )
    })?;
    if corpus.meta.ctx_len == 0 {
        return Err(format!(
            "NN_BAKE_CORPUS entry {path:?} declares meta.ctx_len = 0"
        ));
    }
    if corpus.meta.vocab_size == 0 {
        return Err(format!(
            "NN_BAKE_CORPUS entry {path:?} declares meta.vocab_size = 0"
        ));
    }
    if corpus.rows.is_empty() {
        return Err(format!(
            "NN_BAKE_CORPUS entry {path:?} holds no rows — a source that \
             contributes nothing changes the mixture it was named in"
        ));
    }
    // An id at or past the vocabulary is an out-of-range embedding
    // lookup several layers away from the file that carries it.
    let vocab = corpus.meta.vocab_size as u32;
    for (row_idx, row) in corpus.rows.iter().enumerate() {
        if row.is_empty() {
            return Err(format!(
                "NN_BAKE_CORPUS entry {path:?} row {row_idx} is empty"
            ));
        }
        if let Some(id) = row.iter().copied().find(|id| *id >= vocab) {
            return Err(format!(
                "NN_BAKE_CORPUS entry {path:?} row {row_idx} holds token id {id}, \
                 outside the meta.vocab_size {vocab} it declares"
            ));
        }
    }

    // The requirement list and the field it names have to agree in both
    // directions. Either way round, the run would otherwise train on
    // rows that do not mean what the producer wrote and report the
    // numbers of a well-formed one.
    let required = requires_per_row_allowed(path, corpus.meta.requires.as_ref())?;
    let allowed = match (required, corpus.allowed) {
        (true, None) => {
            return Err(format!(
                "NN_BAKE_CORPUS entry {path:?} declares meta.requires [\"per_row_allowed\"] \
                 but carries no top-level \"allowed\" array — the requirement says the rows \
                 are not the whole corpus and the sets it points at are missing"
            ))
        }
        (false, Some(_)) => {
            return Err(format!(
                "NN_BAKE_CORPUS entry {path:?} carries a top-level \"allowed\" array without \
                 listing \"per_row_allowed\" in meta.requires — unannounced, a reader that \
                 does not implement the field trains on the same rows unconstrained and \
                 reports the same numbers"
            ))
        }
        (false, None) => None,
        (true, Some(sparse)) => Some(densify_allowed(path, &sparse, &corpus.rows, vocab)?),
    };

    Ok(LoadedCorpus {
        ctx_len: corpus.meta.ctx_len,
        vocab_size: corpus.meta.vocab_size,
        rows: corpus.rows,
        allowed,
    })
}

/// Load every corpus, check that they agree on the shape fields, and
/// pair each with its condition slot.
fn load_sources(
    paths: &[String],
    conds: Option<&[u32]>,
) -> Result<(Vec<Source>, usize, usize), String> {
    let mut sources = Vec::with_capacity(paths.len());
    let mut shape: Option<(usize, usize, PathBuf)> = None;
    for (idx, raw) in paths.iter().enumerate() {
        let path = PathBuf::from(raw);
        let corpus = load_corpus(&path)?;
        match &shape {
            None => shape = Some((corpus.ctx_len, corpus.vocab_size, path.clone())),
            Some((ctx_len, vocab_size, first)) => {
                if corpus.ctx_len != *ctx_len || corpus.vocab_size != *vocab_size {
                    return Err(format!(
                        "NN_BAKE_CORPUS entries disagree on shape: {first:?} declares \
                         ctx_len {ctx_len} / vocab_size {vocab_size} and {path:?} declares \
                         ctx_len {} / vocab_size {} — one model cannot be trained on both",
                        corpus.ctx_len, corpus.vocab_size
                    ));
                }
            }
        }
        sources.push(Source {
            path,
            rows: corpus.rows,
            allowed: corpus.allowed,
            cond: conds.map(|c| c[idx]),
        });
    }
    // Every source or none: the rows of one run share a dataset, which
    // takes the sets for all of its rows or for none of them. A mixture
    // would have to invent unconstrained sets for the corpus that
    // carries none, which is a statement its producer did not make.
    let with = sources.iter().find(|s| s.allowed.is_some());
    let without = sources.iter().find(|s| s.allowed.is_none());
    if let (Some(with), Some(without)) = (with, without) {
        return Err(format!(
            "NN_BAKE_CORPUS entry {:?} carries per-row allowed-id sets and {:?} does not — \
             one run trains one dataset, and it either has the sets for every row or for none",
            with.path, without.path
        ));
    }
    let (ctx_len, vocab_size, _) = shape.expect("at least one corpus entry");
    Ok((sources, ctx_len, vocab_size))
}

/// Interleave the sources round-robin, returning the merged rows, for
/// each which source it came from, and — when the sources carry them —
/// the allowed-id sets moved with their own rows.
///
/// Sources of unequal length drop out as they are exhausted, so the
/// tail is whatever the largest ones have left. Nothing is duplicated
/// here: duplicating rows to keep the rotation even would change the
/// mixture the caller asked for.
///
/// The allowed-id sets are pushed in the same statement as the row they
/// belong to, for the reason the conditions are: this rotation is the
/// one place a row and its side channel can come apart while every
/// length still agrees.
#[allow(clippy::type_complexity)]
fn interleave(sources: &[Source]) -> (Vec<Vec<u32>>, Vec<usize>, Vec<Vec<Vec<u32>>>) {
    let longest = sources.iter().map(|s| s.rows.len()).max().unwrap_or(0);
    let total: usize = sources.iter().map(|s| s.rows.len()).sum();
    let mut rows = Vec::with_capacity(total);
    let mut owners = Vec::with_capacity(total);
    let mut allowed = Vec::new();
    if sources.iter().any(|s| s.allowed.is_some()) {
        allowed.reserve(total);
    }
    for position in 0..longest {
        for (src_idx, src) in sources.iter().enumerate() {
            if let Some(row) = src.rows.get(position) {
                rows.push(row.clone());
                owners.push(src_idx);
                if let Some(sets) = src.allowed.as_ref() {
                    allowed.push(sets[position].clone());
                }
            }
        }
    }
    (rows, owners, allowed)
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
    let steps = env_required_usize("NN_BAKE_STEPS", "number of optimizer steps")?;
    if steps == 0 {
        return Err("NN_BAKE_STEPS=0 trains nothing".to_string());
    }
    let batch = env_usize("NN_BAKE_BATCH", 32)?;
    if batch == 0 {
        return Err("NN_BAKE_BATCH=0 leaves every step without rows".to_string());
    }
    let lr = env_f64("NN_BAKE_LR", 3e-3)?;
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
    let pad_id = env_u32("NN_BAKE_PAD_ID", 0)?;

    // The two independent uses of the allowed-id sets: as a mask on the
    // loss, and as an input to the model. Either, both, or neither.
    let mask_logits = env_bool("NN_BAKE_MASK_LOGITS", false)?;
    let allowed_input = env_bool("NN_BAKE_ALLOWED_INPUT", false)?;

    // Conditions, when the caller named any. The two lists are paired
    // positionally, so a length disagreement means some corpus would be
    // trained under another corpus's condition.
    let cond_raw = env_str("NN_BAKE_COND")?;
    let cond_slots_raw = env_str("NN_BAKE_COND_SLOTS")?;
    if cond_raw.is_none() && cond_slots_raw.is_some() {
        return Err(
            "NN_BAKE_COND_SLOTS was set without NN_BAKE_COND — a conditioning table \
             with nothing selecting a row from it trains nothing"
                .to_string(),
        );
    }
    if allowed_input && cond_raw.is_some() {
        // Refused here rather than at the model builder, which would say
        // the same thing one corpus read later: `cond_slots` and
        // `allowed_input` have no combined forward pass, so a model
        // carrying both tables would go through one that drops a channel
        // the caller asked for.
        return Err(
            "NN_BAKE_ALLOWED_INPUT was set together with NN_BAKE_COND — the architecture \
             rejects `cond_slots` together with `allowed_input`, because neither forward \
             pass delivers both channels; pick one"
                .to_string(),
        );
    }
    let conds: Option<Vec<u32>> = match &cond_raw {
        None => None,
        Some(raw) => Some(parse_conds(raw, corpus_paths.len())?),
    };
    let cond_slots: Option<usize> = match &conds {
        None => None,
        Some(slots) => {
            let implied = slots.iter().copied().max().unwrap_or(0) as usize + 1;
            match env_usize("NN_BAKE_COND_SLOTS", implied)? {
                n if n < implied => {
                    return Err(format!(
                        "NN_BAKE_COND_SLOTS={n} is smaller than the {implied} slot(s) \
                         NN_BAKE_COND selects"
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

    let (sources, ctx_len, vocab_size) = load_sources(&corpus_paths, conds.as_deref())?;
    for src in &sources {
        eprintln!(
            "[bake] corpus {:?}: rows={} cond={:?} allowed={}",
            src.path,
            src.rows.len(),
            src.cond,
            src.allowed.is_some()
        );
    }

    let (merged_rows, owners, merged_allowed) = interleave(&sources);
    let corpus_rows = merged_rows.len();

    // The rows describe differing numbers of positions — each runs to
    // its own last listed one — and both readers of the sets take one
    // width for the whole batch. Widening with the empty set says
    // nothing new: a position nobody listed was already unconstrained.
    let allowed_width = merged_allowed.iter().map(Vec::len).max().unwrap_or(0);
    let has_allowed = !merged_allowed.is_empty();
    if has_allowed && allowed_width == 0 {
        return Err(
            "the NN_BAKE_CORPUS entries carry \"allowed\" sets that constrain no position of \
             any row — a producer with nothing to say about the run cannot say it here"
                .to_string(),
        );
    }
    let asked_for_sets = match (mask_logits, allowed_input) {
        (true, true) => Some("NN_BAKE_MASK_LOGITS and NN_BAKE_ALLOWED_INPUT"),
        (true, false) => Some("NN_BAKE_MASK_LOGITS"),
        (false, true) => Some("NN_BAKE_ALLOWED_INPUT"),
        (false, false) => None,
    };
    match (has_allowed, asked_for_sets) {
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

    // The dataset is one-pass, so it has to hold at least as many rows
    // as the run consumes: `steps` batches of `batch`, plus one batch of
    // margin. Cycling repeats the merged (already interleaved) order.
    let needed = steps.saturating_mul(batch).saturating_add(batch);
    let dataset_rows = needed.max(corpus_rows);
    let mut rows = Vec::with_capacity(dataset_rows);
    let mut row_conds: Vec<CondIndex> = Vec::new();
    let mut row_allowed: Vec<Vec<Vec<u32>>> = Vec::new();
    let conditioned = matches!((cond_slots, conds.as_ref()), (Some(_), Some(_)));
    if conditioned {
        row_conds.reserve(dataset_rows);
    }
    if has_allowed {
        row_allowed.reserve(dataset_rows);
    }
    for i in 0..dataset_rows {
        let j = i % corpus_rows;
        rows.push(merged_rows[j].clone());
        // Both side channels are read at the same index as the row, in
        // the same iteration, so the cycling cannot shift one against
        // the other.
        if conditioned {
            let slots = cond_slots.expect("conditioned run");
            let slot = sources[owners[j]].cond.expect("conditioned source");
            row_conds.push(CondIndex::new(slot, slots).map_err(|e| {
                format!("NN_BAKE_COND slot {slot} is not a row of a {slots}-slot table ({e})")
            })?);
        }
        if has_allowed {
            let mut sets = merged_allowed[j].clone();
            sets.resize(allowed_width, Vec::new());
            row_allowed.push(sets);
        }
    }

    let opts = DatasetOpts {
        batch_size: batch,
        ctx_len,
        // Off: the rows were interleaved deliberately above, and the
        // dataset refuses a positional condition list once it has
        // re-ordered its rows.
        shuffle: false,
        pad_id,
        text_field: "text".into(),
    };
    let mut dataset = TokenizedDataset::new(rows, opts);
    if !row_conds.is_empty() {
        dataset = dataset
            .with_conditions(row_conds)
            .map_err(|e| format!("attaching the per-row conditions failed: {e}"))?;
    }
    if !row_allowed.is_empty() {
        // This is where a set that excludes the token its own position
        // holds is refused: the loop below could only answer it with a
        // number.
        dataset = dataset
            .with_allowed_ids(row_allowed)
            .map_err(|e| format!("attaching the per-row allowed-id sets failed: {e}"))?;
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
         mask_logits={mask_logits} device={device_name}"
    );
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
    let model = Gpt2Model::new(&cfg, vb).map_err(|e| format!("building the model failed: {e}"))?;

    let ft_cfg = FullFtConfig {
        lr,
        batch_size: batch,
        steps,
        // Fixed by the experiment series this driver serves.
        warmup: 0,
        schedule: ScheduleKind::Constant,
        mask_disallowed_logits: mask_logits,
        ..FullFtConfig::default()
    };
    let lease = Arc::new(TrainingLease::new());
    let loss = CrossEntropyLoss::new();

    eprintln!(
        "[bake] training steps={steps} batch={batch} lr={lr} rows={dataset_rows} \
         (corpus rows {corpus_rows})"
    );
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

    let summary = serde_json::json!({
        "steps": ckpt.step,
        "final_loss": ckpt.train_loss,
        "min_loss": ckpt.metrics.get("min_train_loss"),
        "elapsed_s": elapsed.as_secs_f64(),
        "out": bundle.to_string_lossy(),
        "rows": dataset_rows,
        "corpus_rows": corpus_rows,
        "cond_slots": cond_slots,
        "allowed_positions": has_allowed.then_some(allowed_width),
        "mask_disallowed_logits": mask_logits,
        "allowed_input": allowed_input,
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
