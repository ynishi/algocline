//! Card metadata schema for `alc.nn.card.*`.
//!
//! Mirrors the `[metadata.nn]` TOML block written by the engine bridge
//! (`bridge/nn_card.rs`) when it assembles the Card create payload.
//! Downstream training paths (Full FT / LoRA / Distillation) populate
//! `hyperparams` / `metrics` / `lineage` uniformly through this schema.
//!
//! `hyperparams` and `metrics` are free-form JSON pass-through so trainer
//! subtasks can extend without reshaping this crate.
//!
//! The Card foundation leaves `NnCandleBranch::lora` as `None`. A
//! later LoRA follow-up populates it via the [`NnLoraBranch`]
//! sub-struct without breaking foundation serialization
//! (`skip_serializing_if = "Option::is_none"`).

use std::path::Path;

use candle_core::safetensors::MmapedSafetensors;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use thiserror::Error;

use crate::arch::{Gpt2Config, Gpt2Custom, MoeConfig};
use crate::train::{Checkpoint, FullFtConfig};

/// Content of `[metadata.nn]`.
///
/// `training_path` and `architecture` are required; everything else is
/// optional with a sensible default so trainer subtasks can populate
/// incrementally.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnCardMeta {
    /// Logical model name (user-facing, e.g. `"domain-classifier-v3"`).
    pub name: String,

    /// Runtime backend of record. `"candle"` for v1; `"endpoint"` /
    /// `"hosted"` / `"adapter"` are v2+ carry.
    pub backend: String,

    /// Informational task label (free-form, e.g. `"classification"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,

    /// Architecture preset id (e.g. `"gpt2-medium"`).
    pub architecture: String,

    /// Training path taken: `"full_ft"` / `"lora"` / `"distillation"` /
    /// `"merged"`. Callers can opt in to validation via
    /// [`validate_training_path`]; the field itself remains a free
    /// `String` at deserialisation time so foundation-era cards with
    /// non-listed values still deserialise (mirrors the
    /// [`validate_architecture`] pattern).
    ///
    /// `"merged"` (Layer 4a) marks a bundle produced by
    /// [`crate::merged::export_merged`] — base + LoRA delta
    /// composed into a single safetensors file, loadable as a plain
    /// base with the same `<Arch>Config`.
    pub training_path: String,

    /// Lineage back-references (parent / teacher / data / tokenizer).
    #[serde(default)]
    pub lineage: NnLineage,

    /// Free-form hyperparameter table (trainer subtasks populate).
    ///
    /// Stored as a JSON object; empty object `{}` means "no hyperparams
    /// recorded" and serializes to an empty TOML sub-table.
    #[serde(default = "empty_object")]
    pub hyperparams: Json,

    /// Free-form training metrics (trainer subtasks populate). Same
    /// semantics as [`Self::hyperparams`].
    #[serde(default = "empty_object")]
    pub metrics: Json,

    /// Backend-specific candle branch. Present when
    /// `backend == "candle"` (v1 default); absent for `endpoint` /
    /// `hosted` cards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candle: Option<NnCandleBranch>,
}

impl NnCardMeta {
    /// Whether this Card declares `axis`.
    ///
    /// A Card with no `candle.custom` branch declares neither: that
    /// branch is absent exactly when the run used the reference
    /// architecture, which has no channel tables.
    pub fn declares_channel(&self, axis: ChannelAxis) -> bool {
        axis.is_set_by(
            self.candle
                .as_ref()
                .and_then(|candle| candle.custom.as_ref())
                .map(|custom| &custom.spec),
        )
    }

    /// Check every [`ChannelAxis`] this Card declares against
    /// `tensor_names`, the names actually present in its bundle.
    ///
    /// The comparison runs in **both** directions, and the second one
    /// is the reason this exists. A declared channel whose table is
    /// missing is caught by the loader soon enough *when the config it
    /// rebuilds asks for the tensor* — the config that asks for
    /// nothing is [`Self::verify_channels_consumed`]'s case, not this
    /// one. A table present in the bundle that
    /// the Card does not declare is caught by nothing: the load path
    /// rebuilds the config from the Card, so it never asks for that
    /// tensor, and the model then runs without an input it was trained
    /// with. Nothing fails, and every number afterwards looks ordinary.
    ///
    /// Axes are inspected in [`ChannelAxis::ALL`] order and the first
    /// disagreement is returned, so the verdict is deterministic for a
    /// Card that is wrong about both.
    ///
    /// # Errors
    ///
    /// [`ChannelMismatch::Missing`] or [`ChannelMismatch::Undeclared`]
    /// for the first axis whose declaration and bundle disagree.
    pub fn verify_channel_tensors<S: AsRef<str>>(
        &self,
        tensor_names: &[S],
    ) -> Result<(), ChannelMismatch> {
        for axis in ChannelAxis::ALL {
            let present = tensor_names
                .iter()
                .any(|name| name.as_ref() == axis.tensor());
            match (self.declares_channel(axis), present) {
                (true, false) => return Err(ChannelMismatch::Missing { axis }),
                (false, true) => return Err(ChannelMismatch::Undeclared { axis }),
                (true, true) | (false, false) => {}
            }
        }
        Ok(())
    }

    /// Check every [`ChannelAxis`] this Card declares against
    /// `consumed`, the customization spec of the config a loader has
    /// just built for it (`None` for a config that has none, which
    /// reads no channel at all).
    ///
    /// [`Self::verify_channel_tensors`] compares the declaration with
    /// the bundle; this compares it with the model about to be built,
    /// and a loader needs both. A config rebuilt for a Card can ignore
    /// the declaration outright — a named architecture pins its own
    /// shape and never consults the shape block — and then the table
    /// sits in the bundle unread, which is the same silence
    /// [`ChannelMismatch::Undeclared`] is about, reached from the
    /// config side instead of the Card side.
    ///
    /// Only the declared ⊆ consumed direction is checked. A config that
    /// reads a channel the Card does not declare cannot come from that
    /// Card's shape block, so it would be the loader's own invention
    /// rather than a disagreement the Card can be judged on.
    ///
    /// # Errors
    ///
    /// [`ChannelMismatch::NotConsumed`] for the first declared axis the
    /// built config does not read.
    pub fn verify_channels_consumed(
        &self,
        consumed: Option<&Gpt2Custom>,
    ) -> Result<(), ChannelMismatch> {
        for axis in ChannelAxis::ALL {
            if self.declares_channel(axis) && !axis.is_set_by(consumed) {
                return Err(ChannelMismatch::NotConsumed { axis });
            }
        }
        Ok(())
    }

    /// [`Self::verify_channel_tensors`] against the bundle stored at
    /// `path`.
    ///
    /// This is the load-time entry point: resolve the Card's
    /// `candle.bundle_ref` to a safetensors path, call this, and only
    /// then build the model. Only the file header is read, so the cost
    /// does not scale with the weights.
    ///
    /// # Errors
    ///
    /// The [`Self::verify_channel_tensors`] verdicts, plus
    /// [`ChannelMismatch::BundleUnreadable`] when the bundle cannot be
    /// opened — a check that could not run is reported rather than
    /// passed.
    pub fn verify_channel_tensors_in_bundle(&self, path: &Path) -> Result<(), ChannelMismatch> {
        let names = bundle_tensor_names(path)?;
        self.verify_channel_tensors(&names)
    }
}

/// Content of `[metadata.nn.lineage]`.
///
/// Every field is optional so a fresh full-FT card (no parent, no
/// teacher) can omit the block entirely without breaking the schema.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct NnLineage {
    /// Prior card id when this model iterates on a predecessor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,

    /// Distillation teacher card ref (e.g. `"cards/haiku-run-042"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<String>,

    /// Training data card ref (e.g. `"cards/prompts-corpus-v1"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_data: Option<String>,

    /// Tokenizer preset id (e.g. `"gpt2"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
}

/// Content of `[metadata.nn.candle]`.
///
/// `bundle_ref` is required and always equal to `"nn/<card_id>"` per
/// design §5 (1:1 mapping between Card id and safetensors bundle name).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnCandleBranch {
    /// Bundle reference `"nn/<card_id>"` — the engine bridge fills this
    /// deterministically from the Card id at save time.
    pub bundle_ref: String,

    /// Device (e.g. `"cuda:0"` / `"cpu"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,

    /// Weights precision (`"bf16"` / `"fp16"` / `"fp32"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// LoRA branch — populated by the LoRA follow-up. The Card
    /// foundation leaves this absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<NnLoraBranch>,

    /// Custom-architecture branch — present only for cards trained
    /// from a `Gpt2Config` that set `custom`. Absent (the common case)
    /// means the bundle carries the GPT-2 reference architecture and
    /// the load path can rebuild the config from the architecture
    /// preset alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<NnCustomBranch>,
}

/// Content of `[metadata.nn.candle.custom]`.
///
/// A preset id such as `"gpt2-custom"` does not pin a shape, so
/// custom-variant cards record their full shape here: the config
/// scalars (`vocab` / `ctx` / `layers` / `heads` / `dim`) plus the
/// [`Gpt2Custom`] spec and the optional MoE block. This is what lets
/// `load_handle` rebuild the exact `Gpt2Config` the run was trained
/// with — without it the safetensors bundle's Var set (which depends
/// on `spec.act` / `spec.pos` / `spec.untied_head` / ...) could not be
/// matched by any config the loader guesses.
///
/// The `dtype` / `device` the weights are read back at stay on the
/// parent [`NnCandleBranch`]; they are load-time choices, not part of
/// the trained shape.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnCustomBranch {
    /// Vocabulary size the `wte` / LM head were trained at.
    pub vocab: usize,

    /// Maximum context length (also the `wpe` size when
    /// `spec.pos == PosKind::Learned`).
    pub ctx: usize,

    /// Number of transformer blocks.
    pub layers: usize,

    /// Number of attention heads.
    pub heads: usize,

    /// Model hidden size.
    pub dim: usize,

    /// Architecture customization spec (activation / norm kind +
    /// placement / residual topology / MLP ratio / position / GQA /
    /// sliding window / untied head / slot conditioning / allowed-id
    /// input).
    ///
    /// # The two axes that are also a claim about the bundle
    ///
    /// Most axes here only shape how the recorded tensors are *used*.
    /// [`Gpt2Custom::cond_slots`] and [`Gpt2Custom::allowed_input`] are
    /// different: each decides whether the bundle carries a table of
    /// its own ([`crate::arch::COND_TABLE_TENSOR`],
    /// [`crate::arch::ALLOWED_TABLE_TENSOR`]), so what this branch
    /// records about them is checkable against the weights — and is
    /// checked, by [`NnCardMeta::verify_channel_tensors`], which the
    /// load path is expected to call. See that method for why the
    /// undeclared direction matters as much as the missing one.
    #[serde(default)]
    pub spec: Gpt2Custom,

    /// Dense-MoE block, when the run combined `custom` with MoE.
    /// Absent means the stock (possibly customized) dense MLP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<NnMoeBranch>,
}

impl NnCustomBranch {
    /// Project a trained [`Gpt2Config`] onto the Card branch, or
    /// `None` when the config is the GPT-2 reference architecture
    /// (`custom: None`).
    ///
    /// `None` is what keeps a stock preset's metadata from growing a
    /// shape block it does not need: `architecture` (e.g.
    /// `"gpt2-medium"`) already pins the shape for those, and the load
    /// path rebuilds them through [`Gpt2Config::from_variant`].
    ///
    /// `eps` is deliberately not recorded — it is not reachable from
    /// the Lua `custom` opts table, so every custom config carries the
    /// [`Gpt2Config::tiny`] default and the load path can restate it.
    /// `dtype` / `device` are load-time choices and live on
    /// [`NnCandleBranch`] instead.
    pub fn from_gpt2_config(cfg: &Gpt2Config) -> Option<Self> {
        let spec = cfg.custom.as_ref()?;
        Some(Self {
            vocab: cfg.vocab,
            ctx: cfg.ctx,
            layers: cfg.layers,
            heads: cfg.heads,
            dim: cfg.dim,
            spec: spec.clone(),
            moe: cfg.moe.as_ref().map(NnMoeBranch::from),
        })
    }
}

/// An optional input channel a custom-architecture run can be trained
/// with.
///
/// Each channel adds a table of its own to the safetensors bundle, so
/// a Card that declares one is making a claim about the tensors sitting
/// beside it — which is what makes
/// [`NnCardMeta::verify_channel_tensors`] possible at all. Axes that
/// only change how existing tensors are used (activation, norm kind,
/// residual topology, ...) leave no such trace and are not represented
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAxis {
    /// Slot conditioning: [`Gpt2Custom::cond_slots`], stored as
    /// [`crate::arch::COND_TABLE_TENSOR`].
    Conditioning,
    /// Allowed-id input: [`Gpt2Custom::allowed_input`], stored as
    /// [`crate::arch::ALLOWED_TABLE_TENSOR`].
    AllowedInput,
}

impl ChannelAxis {
    /// Every channel axis, in the order
    /// [`NnCardMeta::verify_channel_tensors`] inspects them.
    ///
    /// Iterating this rather than listing the axes at each call site is
    /// what makes "a new channel is checked by default" true: a variant
    /// added here reaches the verifier without the verifier changing.
    pub const ALL: [ChannelAxis; 2] = [ChannelAxis::Conditioning, ChannelAxis::AllowedInput];

    /// Tensor this channel's table is stored under in the bundle.
    pub fn tensor(self) -> &'static str {
        match self {
            ChannelAxis::Conditioning => crate::arch::COND_TABLE_TENSOR,
            ChannelAxis::AllowedInput => crate::arch::ALLOWED_TABLE_TENSOR,
        }
    }

    /// Whether `spec` turns this channel on.
    ///
    /// `None` — a config or a Card with no customization spec at all —
    /// turns none of them on: the reference architecture has no channel
    /// tables.
    pub fn is_set_by(self, spec: Option<&Gpt2Custom>) -> bool {
        let Some(spec) = spec else {
            return false;
        };
        match self {
            ChannelAxis::Conditioning => spec.cond_slots.is_some(),
            ChannelAxis::AllowedInput => spec.allowed_input,
        }
    }

    /// Field of `candle.custom.spec` that declares this channel.
    ///
    /// Returned as the serialized spelling, so an error message names
    /// the key a caller would edit rather than a Rust identifier.
    pub fn spec_field(self) -> &'static str {
        match self {
            ChannelAxis::Conditioning => "cond_slots",
            ChannelAxis::AllowedInput => "allowed_input",
        }
    }
}

impl std::fmt::Display for ChannelAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.spec_field())
    }
}

/// A Card's declared channels and the tensors in its bundle disagree.
///
/// Carries the axis rather than the Card id or the bundle path: the
/// caller knows what it was verifying, and both entry points are
/// per-Card. [`ChannelMismatch::BundleUnreadable`] is the exception —
/// there the path is the whole content of the failure.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChannelMismatch {
    /// The Card declares a channel and the bundle has no table for it.
    ///
    /// Loading anyway would build a model with a randomly initialised
    /// table in place of a trained one, and nothing downstream would
    /// look unusual.
    #[error(
        "card declares `{}` but the bundle carries no `{}`; the channel it claims to have been \
         trained with is not in these weights",
        .axis.spec_field(),
        .axis.tensor(),
    )]
    Missing {
        /// Channel the Card declared.
        axis: ChannelAxis,
    },

    /// The bundle carries a channel table the Card does not declare.
    ///
    /// This is the direction that fails quietly when it is not checked:
    /// the load path rebuilds the config from the Card, so an
    /// undeclared table is simply never asked for. The model then runs
    /// without an input it was trained with — well-formed output, every
    /// number afterwards ordinary, and no error anywhere.
    #[error(
        "the bundle carries `{}` but the card does not declare `{}`; loading it would drop a \
         channel the run trained with and every number afterwards would still look ordinary",
        .axis.tensor(),
        .axis.spec_field(),
    )]
    Undeclared {
        /// Channel the bundle carries.
        axis: ChannelAxis,
    },

    /// The Card declares a channel the config rebuilt for it does not
    /// read.
    ///
    /// Not a claim about the weights: the bundle may well carry the
    /// table. The model is the problem — the config asks for no such
    /// input, so the table is never read and the channel is dropped
    /// exactly as quietly as in [`ChannelMismatch::Undeclared`]. The
    /// pair that reaches this is a Card whose architecture pins its own
    /// shape while its shape block declares a channel; no trainer
    /// writes that combination, so such a Card was hand-edited or
    /// written by a foreign pipeline.
    #[error(
        "card declares `{}` but the architecture it names rebuilds a model that never reads it; \
         loading it would drop a channel the run trained with and every number afterwards would \
         still look ordinary",
        .axis.spec_field(),
    )]
    NotConsumed {
        /// Channel the Card declared.
        axis: ChannelAxis,
    },

    /// The bundle could not be opened, so the declared channels could
    /// not be checked against anything.
    ///
    /// Distinct from the two disagreements: nothing is known here, and
    /// treating "could not look" as "nothing to report" is what the
    /// check exists to prevent.
    #[error("bundle {path} could not be read to verify its channel tables: {message}")]
    BundleUnreadable {
        /// Bundle that could not be read.
        path: String,
        /// Reason reported by the safetensors reader.
        message: String,
    },
}

/// Tensor names in the safetensors bundle at `path`.
///
/// Reads the header — a name-to-descriptor map at the front of the file
/// — and materialises no tensor. Deliberately not the reader models
/// load through: a `VarBuilder` answers "is the tensor I asked for
/// here", which by construction cannot see a tensor nobody asked for,
/// and that is exactly the case
/// [`ChannelMismatch::Undeclared`] is about.
fn bundle_tensor_names(path: &Path) -> Result<Vec<String>, ChannelMismatch> {
    // SAFETY: the mapping is unsound if the file is truncated while it
    // is alive. It is created and dropped inside this call, and the
    // bundles this reads are written once and then only read. The same
    // obligation `train::ckpt::restore_into` states applies to the
    // caller.
    let file =
        unsafe { MmapedSafetensors::new(path) }.map_err(|e| ChannelMismatch::BundleUnreadable {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
    Ok(file.tensors().into_iter().map(|(name, _)| name).collect())
}

/// Content of `[metadata.nn.candle.custom.moe]` — the projection of
/// [`crate::arch::MoeConfig`] onto the Card.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnMoeBranch {
    /// Number of experts per block.
    pub n_experts: usize,

    /// Number of experts a token's output is mixed from.
    pub top_k: usize,

    /// Load-balancing loss coefficient.
    pub alpha: f64,
}

impl From<&MoeConfig> for NnMoeBranch {
    /// Records every [`MoeConfig`] field: the three are the whole
    /// config, so a Card that carries this branch describes the MoE
    /// block completely.
    fn from(cfg: &MoeConfig) -> Self {
        Self {
            n_experts: cfg.n_experts,
            top_k: cfg.top_k,
            alpha: cfg.alpha,
        }
    }
}

/// Content of `[metadata.nn.candle.lora]`.
///
/// Present only when `training_path == "lora"`. `rank` / `alpha` /
/// `base_bundle_ref` are always populated by the trainer; the extra
/// `target_modules` / `dropout` / `delta_path` fields carry defaults
/// for backwards compatibility with cards written before ST-d landed
/// (a foundation-era card without these fields still deserializes,
/// but `alc.nn.card.load_gpt2` errors when `delta_path` is missing —
/// the load path cannot locate the delta without it).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NnLoraBranch {
    /// Low-rank decomposition rank (typical: 8 / 16 / 32).
    pub rank: u32,

    /// LoRA scaling factor (`alpha / rank` becomes the effective LR
    /// scaler applied to the delta).
    pub alpha: u32,

    /// Bundle reference for the frozen base model
    /// (e.g. `"nn/base-gpt2-medium"`).
    pub base_bundle_ref: String,

    /// LoRA target modules (subset of
    /// `["q_proj", "k_proj", "v_proj", "o_proj", "up", "down"]`).
    /// Recorded so the load-with-merge path can rebuild the same
    /// `LoraConfig` that produced the delta and wrap the base model
    /// identically.
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,

    /// LoRA dropout probability applied during training. Currently
    /// held for provenance only — the shipped [`crate::arch::LoraLinear`]
    /// forward does not apply dropout at inference; the field is
    /// preserved so a future dropout-at-train-only variant can be
    /// distinguished from cards trained without it.
    #[serde(default)]
    pub dropout: f32,

    /// Absolute path of the delta safetensors file emitted by
    /// [`crate::train::run_lora_ft`] at
    /// `<ckpt_dir>/nn/lora-<ckpt_prefix>.safetensors`. Recorded so
    /// `alc.nn.card.load_gpt2` can locate the delta without knowing
    /// the caller's `ckpt_dir` / `ckpt_prefix` conventions.
    ///
    /// Absent for hand-authored cards; the load path errors loudly
    /// when a lora card lacks `delta_path` rather than guessing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_path: Option<String>,
}

/// Default target modules when a LoRA card was written before ST-d
/// added the field. Matches
/// [`crate::arch::LoraConfig::default_targets`].
fn default_target_modules() -> Vec<String> {
    vec![
        "q_proj".into(),
        "k_proj".into(),
        "v_proj".into(),
        "o_proj".into(),
        "up".into(),
        "down".into(),
    ]
}

/// Default value for `hyperparams` / `metrics` — an empty JSON object.
fn empty_object() -> Json {
    Json::Object(serde_json::Map::new())
}

/// Architecture family prefixes accepted for [`NnCardMeta::architecture`].
///
/// Values written by the engine bridge use the `<family>-<variant>` shape
/// (e.g. `"gpt2-medium"`, `"llama-3.2-1b"`, `"tinyllama-1.1b"`). At save
/// time the bridge validates the prefix against this list so a typo does
/// not silently write an unloadable Card, and downstream loaders can
/// route by family prefix without a second lookup table.
///
/// Extend this list when a new trainable or inference-only architecture
/// lands; the corresponding `alc.nn.preset.<family>` binding is expected
/// to populate the same prefix.
pub const SUPPORTED_ARCHITECTURE_FAMILIES: &[&str] =
    &["gpt2", "llama", "tinyllama", "qwen2", "phi", "gemma"];

/// Training paths accepted for [`NnCardMeta::training_path`].
///
/// Values written by the engine bridge or by
/// [`crate::merged::export_merged`] must match one of these strings.
/// Extend this list when a new training-path is added; the
/// corresponding sub-branch (if any) is registered alongside on
/// [`NnCandleBranch`]. `"merged"` (Layer 4a) has no sub-branch —
/// its provenance rides on `NnLineage.parent` per the Q0
/// Model-side struct + projection pattern.
pub const SUPPORTED_TRAINING_PATHS: &[&str] = &["full_ft", "lora", "distillation", "merged"];

/// Validate that `training_path` is one of
/// [`SUPPORTED_TRAINING_PATHS`].
///
/// Callers opt in to this check (mirrors [`validate_architecture`]);
/// [`NnCardMeta`] itself deserialises any string so foundation-era
/// cards with non-listed values still round-trip.
pub fn validate_training_path(training_path: &str) -> Result<(), String> {
    if training_path.is_empty() {
        return Err("training_path must not be empty".into());
    }
    for accepted in SUPPORTED_TRAINING_PATHS {
        if training_path == *accepted {
            return Ok(());
        }
    }
    Err(format!(
        "unknown training_path {training_path:?} (expected one of {SUPPORTED_TRAINING_PATHS:?})"
    ))
}

/// Validate that `arch` starts with a known family prefix from
/// [`SUPPORTED_ARCHITECTURE_FAMILIES`].
///
/// Accepts either the bare family name (`"gpt2"`) or the
/// `<family>-<variant>` form (`"gpt2-medium"`). The prefix must be
/// followed by end-of-string or `-`; a longer identifier that happens to
/// start with a family name (`"gpt2experimental"`) is rejected so the
/// namespace stays partitioned.
pub fn validate_architecture(arch: &str) -> Result<(), String> {
    if arch.is_empty() {
        return Err("architecture must not be empty".into());
    }
    for family in SUPPORTED_ARCHITECTURE_FAMILIES {
        if arch == *family {
            return Ok(());
        }
        if let Some(rest) = arch.strip_prefix(family) {
            if rest.starts_with('-') {
                return Ok(());
            }
        }
    }
    Err(format!(
        "unknown architecture family for {arch:?} (expected one of {SUPPORTED_ARCHITECTURE_FAMILIES:?}, \
         optionally followed by '-<variant>')"
    ))
}

/// Character set accepted for [`CardId`] values.
///
/// Matches the stricter of the two storage layers a Card id must
/// satisfy: `FileCardStore::validate_name` (no `/` / `\` / `..` /
/// `\0`) and `FsStore`'s `[A-Za-z0-9_.-]` alphabet (the safetensors
/// bundle filename is derived from the id 1:1).
fn is_valid_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-'
}

/// Mint a unique, filesystem-safe stem from a free-form `name`.
///
/// Format: `<sanitized_name>_<epoch_us>`. Non `[A-Za-z0-9_-]`
/// characters collapse to `_`; an all-invalid (or empty) name falls
/// back to `"nn"`. The microsecond suffix keeps rapid successive
/// calls unique without pulling in a UUID crate; a clock-skew corner
/// (`SystemTime` < `UNIX_EPOCH`) collapses to epoch zero and any id
/// collision then surfaces loudly through the card store's
/// immutable-card guard rather than silently.
///
/// This is the single implementation behind both [`CardId::mint`]
/// (Card ids) and the engine bridge's checkpoint filename stems —
/// the two use the same collision-avoidance format but only the
/// former is a Card identifier.
pub fn unique_stem(name: &str) -> String {
    let sanitized = sanitize_stem(name);
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{sanitized}_{}{:06}", d.as_secs(), d.subsec_micros())
}

/// Collapse a free-form name into the `[A-Za-z0-9_-]` stem alphabet
/// (no uniqueness suffix — see [`unique_stem`] for that). An
/// all-invalid or empty input falls back to `"nn"`.
pub fn sanitize_stem(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "nn".into()
    } else {
        out
    }
}

/// Logical `"nn/<stem>"` bundle reference.
///
/// The single place the `nn/` prefix is attached. Card ids derive
/// their reference via [`CardId::bundle_ref`] (which delegates here);
/// non-Card base bundles (e.g. the pretrained `"gpt2-medium"`
/// weights referenced by `NnLoraBranch.base_bundle_ref`) use this
/// function directly.
pub fn bundle_ref_for(stem: &str) -> String {
    format!("nn/{stem}")
}

/// Validated identifier of an `nn_model` Card.
///
/// A `CardId` can only be obtained through [`CardId::mint`] (fresh id
/// from a user-facing name) or [`CardId::parse`] / `TryFrom<&str>`
/// (validated from caller input), so any value of this type is safe
/// to use both as a card-store key and as a safetensors bundle
/// filename. The logical bundle reference is *derived* from the id
/// via [`CardId::bundle_ref`] — `"nn/<id>"` is not stored anywhere it
/// could drift.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CardId(String);

impl CardId {
    /// Mint a fresh id from a user-facing `name` (see [`unique_stem`]).
    pub fn mint(name: &str) -> Self {
        Self(unique_stem(name))
    }

    /// Validate `s` as a Card id.
    ///
    /// Rejects empty strings, path-traversal stand-ins (`"."` /
    /// `".."`), and any character outside `[A-Za-z0-9_.-]`. Every id
    /// produced by [`CardId::mint`] passes; hand-authored ids that
    /// could never resolve to an on-disk bundle are refused up front
    /// with the offending character named.
    pub fn parse(s: &str) -> Result<Self, String> {
        if s.is_empty() {
            return Err("card_id must not be empty".into());
        }
        if s == "." || s == ".." {
            return Err(format!("card_id {s:?} is not a valid identifier"));
        }
        if let Some(bad) = s.chars().find(|c| !is_valid_id_char(*c)) {
            return Err(format!(
                "card_id {s:?} contains invalid character {bad:?} \
                 (allowed: A-Z a-z 0-9 '_' '.' '-')"
            ));
        }
        Ok(Self(s.to_string()))
    }

    /// The id as a plain string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the id, returning the inner `String`.
    pub fn into_string(self) -> String {
        self.0
    }

    /// Logical bundle reference for this Card: `"nn/<id>"`.
    ///
    /// Callers must never assemble the string by hand (design §5:
    /// 1:1 mapping between Card id and safetensors bundle name);
    /// see [`bundle_ref_for`].
    pub fn bundle_ref(&self) -> String {
        bundle_ref_for(&self.0)
    }
}

impl TryFrom<&str> for CardId {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl AsRef<str> for CardId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which training path produced a Card, plus the path-specific
/// payload that rides on the Card.
///
/// Used by [`NnModelCard::from_training`]; the merge path has its own
/// constructor ([`NnModelCard::from_merge`]) because a merged bundle
/// is projected from [`crate::merged::MergedProvenance`] rather than
/// from a live training run.
#[derive(Debug, Clone)]
pub enum TrainingPath {
    /// Full fine-tune — no sub-branch.
    FullFt,
    /// LoRA fine-tune — carries the delta provenance branch.
    Lora(NnLoraBranch),
    /// Distillation — records which loss variant drove the run.
    Distillation {
        /// Loss selector recorded in `hyperparams.loss_kind`
        /// (e.g. `"ce"`).
        loss_kind: String,
    },
}

impl TrainingPath {
    /// The `training_path` string recorded on the Card.
    pub fn as_str(&self) -> &'static str {
        match self {
            TrainingPath::FullFt => "full_ft",
            TrainingPath::Lora(_) => "lora",
            TrainingPath::Distillation { .. } => "distillation",
        }
    }
}

/// Aggregate tying a [`CardId`] to its [`NnCardMeta`].
///
/// Construction is the invariant gate: every constructor routes
/// through [`NnModelCard::new`], which enforces
///
/// 1. `architecture` matches a known family
///    ([`validate_architecture`]),
/// 2. `training_path` is a supported value
///    ([`validate_training_path`]),
/// 3. when a candle branch is present, its `bundle_ref` equals the
///    value derived from the id ([`CardId::bundle_ref`]).
///
/// A persisted Card therefore cannot carry a `bundle_ref` that
/// diverges from its id — the assert-after-the-fact checks the
/// engine bridge used to scatter across call sites are replaced by
/// this single construction-time guarantee.
#[derive(Debug, Clone)]
pub struct NnModelCard {
    id: CardId,
    meta: NnCardMeta,
}

impl NnModelCard {
    /// Assemble an aggregate from parts, enforcing the Card
    /// invariants (see type-level docs). This is the only door —
    /// `id` / `meta` are private so a shape that violates the
    /// invariants cannot exist.
    pub fn new(id: CardId, meta: NnCardMeta) -> Result<Self, String> {
        validate_architecture(&meta.architecture)?;
        validate_training_path(&meta.training_path)?;
        if let Some(candle) = &meta.candle {
            let expected = id.bundle_ref();
            if candle.bundle_ref != expected {
                return Err(format!(
                    "bundle_ref {:?} does not match card_id {:?} (expected {expected:?})",
                    candle.bundle_ref,
                    id.as_str()
                ));
            }
        }
        Ok(Self { id, meta })
    }

    /// Build a Card from a finished training run.
    ///
    /// This is the single Checkpoint ↔ Card contact point: it derives
    /// the candle branch's `bundle_ref` from `id` and records
    /// `hyperparams` / `metrics` from the typed [`FullFtConfig`] /
    /// [`Checkpoint`] instead of caller-assembled JSON.
    ///
    /// `id` is caller-minted (via [`CardId::mint`]) rather than
    /// minted here because the full-FT / distillation loops use the
    /// id as the checkpoint filename stem *before* the run finishes —
    /// the id must exist first so `<ckpt_dir>/<id>.safetensors` and
    /// the Card stay 1:1.
    ///
    /// `custom` records the architecture shape for runs whose config
    /// set `Gpt2Config::custom`; pass `None` for reference-architecture
    /// runs, whose shape is already implied by `architecture`.
    ///
    /// `device` / `dtype` record the training-time device and dtype
    /// string ("cpu" / "cuda" / "metal", "f32" / "f16" / "bf16", …)
    /// so `load_handle` restores the same target instead of silently
    /// falling back to the arch default (which is CPU / f32 for
    /// `Gpt2Config::tiny()`). Pass the same `None` pair the older
    /// signature implied when the caller has no handle-level target
    /// to record.
    #[allow(clippy::too_many_arguments)]
    pub fn from_training(
        id: CardId,
        name: &str,
        architecture: String,
        path: TrainingPath,
        ckpt: &Checkpoint,
        cfg: &FullFtConfig,
        custom: Option<NnCustomBranch>,
        device: Option<String>,
        dtype: Option<String>,
    ) -> Result<Self, String> {
        let training_path = path.as_str().to_string();
        let (lora, loss_kind) = match path {
            TrainingPath::FullFt => (None, None),
            TrainingPath::Lora(branch) => (Some(branch), None),
            TrainingPath::Distillation { loss_kind } => (None, Some(loss_kind)),
        };

        let mut hyperparams = serde_json::Map::new();
        hyperparams.insert("lr".into(), serde_json::json!(cfg.lr));
        hyperparams.insert("batch".into(), serde_json::json!(cfg.batch_size));
        hyperparams.insert("steps".into(), serde_json::json!(cfg.steps));
        hyperparams.insert("warmup".into(), serde_json::json!(cfg.warmup));
        if let Some(loss_kind) = loss_kind {
            hyperparams.insert("loss_kind".into(), serde_json::json!(loss_kind));
        }

        let mut metrics = serde_json::Map::new();
        metrics.insert("train_loss".into(), serde_json::json!(ckpt.train_loss));
        metrics.insert("step".into(), serde_json::json!(ckpt.step));

        let candle = NnCandleBranch {
            bundle_ref: id.bundle_ref(),
            device,
            dtype,
            lora,
            custom,
        };

        let meta = NnCardMeta {
            name: name.to_string(),
            backend: "candle".into(),
            task: None,
            architecture,
            training_path,
            lineage: NnLineage::default(),
            hyperparams: Json::Object(hyperparams),
            metrics: Json::Object(metrics),
            candle: Some(candle),
        };
        Self::new(id, meta)
    }

    /// Build a Card for a merged inference bundle.
    ///
    /// `meta` is the projection returned by
    /// [`crate::merged::MergedProvenance::to_card_meta`] (whose
    /// `bundle_ref` the caller derived from `id` when constructing
    /// the provenance); `name` overrides the projection's default
    /// (bundle file stem) with the user-visible Card name. The
    /// bundle_ref ↔ id coherence is re-checked by [`Self::new`].
    pub fn from_merge(id: CardId, name: String, mut meta: NnCardMeta) -> Result<Self, String> {
        meta.name = name;
        Self::new(id, meta)
    }

    /// The validated Card id.
    pub fn id(&self) -> &CardId {
        &self.id
    }

    /// The typed `[metadata.nn]` block.
    pub fn meta(&self) -> &NnCardMeta {
        &self.meta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_ft_meta_roundtrips_through_json() {
        let meta = NnCardMeta {
            name: "domain-classifier-v3".into(),
            backend: "candle".into(),
            task: Some("classification".into()),
            architecture: "gpt2-medium".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage {
                tokenizer: Some("gpt2".into()),
                ..Default::default()
            },
            hyperparams: serde_json::json!({ "lr": 3e-4, "steps": 20000 }),
            metrics: serde_json::json!({ "val_loss_final": 0.51 }),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/abc123".into(),
                device: Some("cuda:0".into()),
                dtype: Some("bf16".into()),
                lora: None,
                custom: None,
            }),
        };

        let json = serde_json::to_value(&meta).expect("serialize");
        let back: NnCardMeta = serde_json::from_value(json.clone()).expect("deserialize");

        assert_eq!(back.name, meta.name);
        assert_eq!(back.training_path, "full_ft");
        assert_eq!(back.candle.as_ref().unwrap().bundle_ref, "nn/abc123");
        assert!(back.candle.as_ref().unwrap().lora.is_none());
        assert_eq!(back.lineage.tokenizer.as_deref(), Some("gpt2"));

        // Round-trip preserves every field verbatim.
        let json2 = serde_json::to_value(&back).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn lora_branch_serializes_when_present() {
        let meta = NnCardMeta {
            name: "student-lora".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "lora".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/student-lora".into(),
                device: None,
                dtype: None,
                lora: Some(NnLoraBranch {
                    rank: 16,
                    alpha: 32,
                    base_bundle_ref: "nn/base-gpt2-medium".into(),
                    target_modules: default_target_modules(),
                    dropout: 0.0,
                    delta_path: Some(
                        "/var/algocline/nn/ckpt/nn/lora-student-lora.safetensors".into(),
                    ),
                }),
                custom: None,
            }),
        };

        let json = serde_json::to_value(&meta).expect("serialize");
        let candle = json.get("candle").expect("candle branch present");
        let lora = candle.get("lora").expect("lora sub-object present");
        assert_eq!(lora.get("rank"), Some(&serde_json::json!(16)));
        assert_eq!(lora.get("alpha"), Some(&serde_json::json!(32)));
        assert_eq!(
            lora.get("base_bundle_ref"),
            Some(&serde_json::json!("nn/base-gpt2-medium"))
        );
        // Post-ST-d extension: target_modules + dropout serialize
        // alongside the trio.
        assert_eq!(
            lora.get("target_modules"),
            Some(&serde_json::json!(default_target_modules()))
        );
        assert_eq!(lora.get("dropout"), Some(&serde_json::json!(0.0)));
        assert_eq!(
            lora.get("delta_path"),
            Some(&serde_json::json!(
                "/var/algocline/nn/ckpt/nn/lora-student-lora.safetensors"
            ))
        );
    }

    /// A pre-ST-d card (only `rank` / `alpha` / `base_bundle_ref` in
    /// the lora sub-table) must still deserialize. `target_modules`
    /// falls back to the canonical six, `dropout` to 0.0, and
    /// `delta_path` to `None` — the load path then errors when it
    /// tries to locate the delta.
    #[test]
    fn deserialize_backwards_compat_lora_without_new_fields() {
        let legacy = serde_json::json!({
            "name": "legacy-lora",
            "backend": "candle",
            "architecture": "gpt2-medium",
            "training_path": "lora",
            "candle": {
                "bundle_ref": "nn/legacy-lora",
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "base_bundle_ref": "nn/base-gpt2-medium",
                }
            }
        });
        let meta: NnCardMeta = serde_json::from_value(legacy).expect("legacy lora deserialize");
        let lora = meta
            .candle
            .as_ref()
            .and_then(|c| c.lora.as_ref())
            .expect("lora branch present");
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.alpha, 16);
        assert_eq!(lora.target_modules, default_target_modules());
        assert_eq!(lora.dropout, 0.0);
        assert!(
            lora.delta_path.is_none(),
            "delta_path defaults to None for pre-ST-d cards"
        );
    }

    #[test]
    fn absent_lora_is_omitted_not_null() {
        let meta = NnCardMeta {
            name: "no-lora".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/x".into(),
                device: None,
                dtype: None,
                lora: None,
                custom: None,
            }),
        };
        let json = serde_json::to_value(&meta).expect("serialize");
        let candle = json.get("candle").expect("candle branch");
        assert!(
            candle.get("lora").is_none(),
            "lora must be omitted (skip_serializing_if), got: {candle}"
        );
        assert!(
            candle.get("custom").is_none(),
            "custom must be omitted (skip_serializing_if), got: {candle}"
        );
    }

    fn sample_custom_branch() -> NnCustomBranch {
        NnCustomBranch {
            vocab: 64,
            ctx: 16,
            layers: 2,
            heads: 4,
            dim: 32,
            spec: Gpt2Custom {
                act: crate::arch::Activation::SwiGlu,
                norm: crate::arch::NormKind::RmsNorm,
                residual: crate::arch::ResidualKind::Parallel,
                mlp_ratio: 3,
                placement: crate::arch::NormPlacement::PreLn,
                pos: crate::arch::PosKind::Rope,
                kv_heads: Some(1),
                window: Some(4),
                untied_head: true,
                cond_slots: None,
                allowed_input: false,
            },
            moe: Some(NnMoeBranch {
                n_experts: 4,
                top_k: 2,
                alpha: 0.01,
            }),
        }
    }

    /// A custom-variant card records its full shape so `load_handle`
    /// can rebuild the identical `Gpt2Config`; the whole branch must
    /// survive a JSON round-trip verbatim, non-default axes included.
    #[test]
    fn custom_branch_roundtrips_with_non_default_axes() {
        let meta = NnCardMeta {
            name: "gpt2-custom-run".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-custom".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/gpt2-custom-run".into(),
                device: None,
                dtype: None,
                lora: None,
                custom: Some(sample_custom_branch()),
            }),
        };

        let json = serde_json::to_value(&meta).expect("serialize");
        let custom = json
            .get("candle")
            .and_then(|c| c.get("custom"))
            .expect("custom sub-object present");
        assert_eq!(custom.get("vocab"), Some(&serde_json::json!(64)));
        assert_eq!(custom.get("ctx"), Some(&serde_json::json!(16)));
        assert_eq!(custom.get("layers"), Some(&serde_json::json!(2)));
        assert_eq!(custom.get("heads"), Some(&serde_json::json!(4)));
        assert_eq!(custom.get("dim"), Some(&serde_json::json!(32)));
        // Spec axes use the Lua-facing vocabulary.
        let spec = custom.get("spec").expect("spec sub-object");
        assert_eq!(spec.get("act"), Some(&serde_json::json!("swiglu")));
        assert_eq!(spec.get("norm"), Some(&serde_json::json!("rmsnorm")));
        assert_eq!(spec.get("residual"), Some(&serde_json::json!("parallel")));
        assert_eq!(spec.get("placement"), Some(&serde_json::json!("preln")));
        assert_eq!(spec.get("pos"), Some(&serde_json::json!("rope")));
        assert_eq!(spec.get("mlp_ratio"), Some(&serde_json::json!(3)));
        assert_eq!(spec.get("kv_heads"), Some(&serde_json::json!(1)));
        assert_eq!(spec.get("window"), Some(&serde_json::json!(4)));
        assert_eq!(spec.get("untied_head"), Some(&serde_json::json!(true)));
        let moe = custom.get("moe").expect("moe sub-object");
        assert_eq!(moe.get("n_experts"), Some(&serde_json::json!(4)));
        assert_eq!(moe.get("top_k"), Some(&serde_json::json!(2)));

        let back: NnCardMeta = serde_json::from_value(json.clone()).expect("deserialize");
        let branch = back
            .candle
            .as_ref()
            .and_then(|c| c.custom.as_ref())
            .expect("custom branch present");
        assert_eq!(branch.vocab, 64);
        assert_eq!(branch.spec.act, crate::arch::Activation::SwiGlu);
        assert_eq!(branch.spec.pos, crate::arch::PosKind::Rope);
        assert_eq!(branch.spec.kv_heads, Some(1));
        assert!(branch.spec.untied_head);
        assert_eq!(branch.moe.as_ref().map(|m| m.top_k), Some(2));
        assert_eq!(serde_json::to_value(&back).expect("re-serialize"), json);
    }

    /// A card written before the custom branch existed has no
    /// `candle.custom` key; it must still deserialize, with the branch
    /// absent (meaning "reference architecture").
    #[test]
    fn deserialize_backwards_compat_candle_without_custom() {
        let legacy = serde_json::json!({
            "name": "legacy-full-ft",
            "backend": "candle",
            "architecture": "gpt2-medium",
            "training_path": "full_ft",
            "candle": {
                "bundle_ref": "nn/legacy-full-ft",
                "dtype": "bf16",
            }
        });
        let meta: NnCardMeta = serde_json::from_value(legacy).expect("legacy card deserialize");
        let candle = meta.candle.as_ref().expect("candle branch present");
        assert_eq!(candle.bundle_ref, "nn/legacy-full-ft");
        assert_eq!(candle.dtype.as_deref(), Some("bf16"));
        assert!(
            candle.custom.is_none(),
            "custom defaults to None for pre-custom cards"
        );
    }

    /// A custom spec table that only names one axis must fill the rest
    /// from the reference, so a hand-authored (or older) card is not
    /// rejected for omitting axes.
    #[test]
    fn deserialize_custom_branch_with_partial_spec() {
        let partial = serde_json::json!({
            "name": "partial-custom",
            "backend": "candle",
            "architecture": "gpt2-custom",
            "training_path": "full_ft",
            "candle": {
                "bundle_ref": "nn/partial-custom",
                "custom": {
                    "vocab": 64,
                    "ctx": 16,
                    "layers": 2,
                    "heads": 2,
                    "dim": 32,
                    "spec": { "norm": "rmsnorm" }
                }
            }
        });
        let meta: NnCardMeta = serde_json::from_value(partial).expect("partial custom deserialize");
        let branch = meta
            .candle
            .as_ref()
            .and_then(|c| c.custom.as_ref())
            .expect("custom branch present");
        assert_eq!(branch.spec.norm, crate::arch::NormKind::RmsNorm);
        assert_eq!(branch.spec.act, crate::arch::Activation::Gelu);
        assert_eq!(branch.spec.pos, crate::arch::PosKind::Learned);
        assert_eq!(branch.spec.kv_heads, None);
        assert!(!branch.spec.untied_head);
        assert!(branch.moe.is_none());
    }

    /// The `Gpt2Config` -> Card projection must copy the five shape
    /// scalars and the spec verbatim, and fold `moe` through
    /// `NnMoeBranch::from` — the load path rebuilds the config from
    /// exactly these values, so a dropped field is a silently
    /// different model on reload.
    #[test]
    fn custom_branch_from_gpt2_config_projects_shape_and_moe() {
        let spec = Gpt2Custom {
            act: crate::arch::Activation::Gelu,
            norm: crate::arch::NormKind::RmsNorm,
            pos: crate::arch::PosKind::Rope,
            kv_heads: Some(2),
            untied_head: true,
            ..Gpt2Custom::default()
        };
        let cfg = Gpt2Config {
            layers: 3,
            heads: 4,
            dim: 64,
            ctx: 32,
            vocab: 96,
            custom: Some(spec.clone()),
            moe: Some(MoeConfig {
                n_experts: 4,
                top_k: 2,
                alpha: 0.01,
            }),
            ..Gpt2Config::tiny()
        };
        let branch = NnCustomBranch::from_gpt2_config(&cfg).expect("custom config projects");
        assert_eq!(branch.layers, 3);
        assert_eq!(branch.heads, 4);
        assert_eq!(branch.dim, 64);
        assert_eq!(branch.ctx, 32);
        assert_eq!(branch.vocab, 96);
        assert_eq!(
            serde_json::to_value(&branch.spec).expect("spec json"),
            serde_json::to_value(&spec).expect("spec json")
        );
        let moe = branch.moe.expect("moe branch");
        assert_eq!(moe.n_experts, 4);
        assert_eq!(moe.top_k, 2);
        assert_eq!(moe.alpha, 0.01);
    }

    /// A reference-architecture config must project to `None` so stock
    /// presets keep a lean `candle` branch (`architecture` alone pins
    /// their shape).
    #[test]
    fn custom_branch_from_gpt2_config_is_none_for_reference_config() {
        assert!(NnCustomBranch::from_gpt2_config(&Gpt2Config::tiny()).is_none());
        assert!(NnCustomBranch::from_gpt2_config(&Gpt2Config::medium()).is_none());
    }

    #[test]
    fn from_training_carries_custom_branch() {
        let card = NnModelCard::from_training(
            CardId::mint("custom-run"),
            "custom-run",
            "gpt2-custom".into(),
            TrainingPath::FullFt,
            &test_ckpt(),
            &FullFtConfig::default(),
            Some(sample_custom_branch()),
            None,
            None,
        )
        .expect("from_training");
        let branch = card
            .meta()
            .candle
            .as_ref()
            .and_then(|c| c.custom.as_ref())
            .expect("custom branch");
        assert_eq!(branch.dim, 32);
        assert_eq!(branch.spec.act, crate::arch::Activation::SwiGlu);
    }

    #[test]
    fn deserialize_tolerates_absent_optional_fields() {
        let minimal = serde_json::json!({
            "name": "m",
            "backend": "candle",
            "architecture": "gpt2-medium",
            "training_path": "full_ft",
        });
        let meta: NnCardMeta = serde_json::from_value(minimal).expect("deserialize minimal");
        assert!(meta.task.is_none());
        assert!(meta.candle.is_none());
        assert!(meta.lineage.parent.is_none());
        assert_eq!(meta.hyperparams, empty_object());
        assert_eq!(meta.metrics, empty_object());
    }

    #[test]
    fn validate_architecture_accepts_known_families() {
        for arch in [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-tiny",
            "llama",
            "llama-3.2-1b",
            "tinyllama",
            "tinyllama-1.1b",
            "tinyllama-tiny",
            "qwen2",
            "qwen2-0.5b",
            "phi",
            "phi-1.5",
            "gemma",
            "gemma-2b",
        ] {
            assert!(
                validate_architecture(arch).is_ok(),
                "expected {arch} to be accepted"
            );
        }
    }

    #[test]
    fn validate_architecture_rejects_unknown_and_typos() {
        for arch in ["", "gpt3", "mistral-7b", "gpt2experimental", " gpt2-medium"] {
            assert!(
                validate_architecture(arch).is_err(),
                "expected {arch:?} to be rejected"
            );
        }
    }

    #[test]
    fn card_id_mint_sanitizes_and_parses_back() {
        let id = CardId::mint("my model/v2!");
        assert!(
            id.as_str().starts_with("my_model_v2__"),
            "sanitized prefix expected, got {:?}",
            id.as_str()
        );
        // Every minted id must survive its own validation round-trip.
        let reparsed = CardId::parse(id.as_str()).expect("minted id parses");
        assert_eq!(reparsed, id);
        assert_eq!(id.bundle_ref(), format!("nn/{}", id.as_str()));
    }

    #[test]
    fn card_id_mint_empty_name_falls_back_to_nn() {
        let id = CardId::mint("");
        assert!(id.as_str().starts_with("nn_"), "got {:?}", id.as_str());
    }

    #[test]
    fn card_id_parse_rejects_invalid() {
        for bad in ["", ".", "..", "a/b", "a\\b", "a b", "a\0b", "日本語"] {
            assert!(
                CardId::parse(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn card_id_parse_accepts_store_safe_ids() {
        for ok in ["abc", "a-b_c.d", "model_1785000000123456", "A.B-C_9"] {
            assert!(CardId::parse(ok).is_ok(), "expected {ok:?} to be accepted");
        }
    }

    fn test_ckpt() -> Checkpoint {
        Checkpoint {
            bundle_ref: "x.safetensors".into(),
            step: 42,
            train_loss: 1.25,
            val_loss: None,
            metrics: std::collections::HashMap::new(),
            candidates: Vec::new(),
        }
    }

    #[test]
    fn from_training_full_ft_derives_bundle_ref_and_typed_tables() {
        let card = NnModelCard::from_training(
            CardId::mint("runner"),
            "runner",
            "gpt2-medium".into(),
            TrainingPath::FullFt,
            &test_ckpt(),
            &FullFtConfig::default(),
            None,
            None,
            None,
        )
        .expect("from_training");

        let meta = card.meta();
        assert_eq!(meta.training_path, "full_ft");
        let candle = meta.candle.as_ref().expect("candle branch");
        assert_eq!(candle.bundle_ref, card.id().bundle_ref());
        assert!(candle.lora.is_none());
        assert_eq!(meta.hyperparams.get("lr"), Some(&serde_json::json!(3e-4)));
        assert_eq!(meta.hyperparams.get("batch"), Some(&serde_json::json!(8)));
        assert_eq!(meta.metrics.get("step"), Some(&serde_json::json!(42)));
        assert_eq!(
            meta.metrics.get("train_loss"),
            Some(&serde_json::json!(1.25))
        );
        assert!(meta.hyperparams.get("loss_kind").is_none());
        assert!(
            candle.device.is_none(),
            "None device is preserved as absent"
        );
        assert!(candle.dtype.is_none(), "None dtype is preserved as absent");
    }

    #[test]
    fn from_training_records_device_and_dtype_when_supplied() {
        let card = NnModelCard::from_training(
            CardId::mint("gpu-run"),
            "gpu-run",
            "gpt2-medium".into(),
            TrainingPath::FullFt,
            &test_ckpt(),
            &FullFtConfig::default(),
            None,
            Some("cuda".into()),
            Some("bf16".into()),
        )
        .expect("from_training");
        let candle = card.meta().candle.as_ref().expect("candle branch");
        assert_eq!(candle.device.as_deref(), Some("cuda"));
        assert_eq!(candle.dtype.as_deref(), Some("bf16"));
    }

    #[test]
    fn from_training_distillation_records_loss_kind() {
        let card = NnModelCard::from_training(
            CardId::mint("student"),
            "student",
            "tinyllama-1.1b".into(),
            TrainingPath::Distillation {
                loss_kind: "ce".into(),
            },
            &test_ckpt(),
            &FullFtConfig::default(),
            None,
            None,
            None,
        )
        .expect("from_training");
        assert_eq!(card.meta().training_path, "distillation");
        assert_eq!(
            card.meta().hyperparams.get("loss_kind"),
            Some(&serde_json::json!("ce"))
        );
    }

    #[test]
    fn from_training_lora_carries_branch() {
        let branch = NnLoraBranch {
            rank: 16,
            alpha: 32,
            base_bundle_ref: "nn/gpt2-medium".into(),
            target_modules: default_target_modules(),
            dropout: 0.0,
            delta_path: Some("/tmp/nn/lora-x.safetensors".into()),
        };
        let card = NnModelCard::from_training(
            CardId::mint("adapter"),
            "adapter",
            "gpt2-medium".into(),
            TrainingPath::Lora(branch),
            &test_ckpt(),
            &FullFtConfig::default(),
            None,
            None,
            None,
        )
        .expect("from_training");
        assert_eq!(card.meta().training_path, "lora");
        let lora = card
            .meta()
            .candle
            .as_ref()
            .and_then(|c| c.lora.as_ref())
            .expect("lora branch");
        assert_eq!(lora.rank, 16);
    }

    #[test]
    fn new_rejects_bundle_ref_id_mismatch() {
        let id = CardId::mint("x");
        let meta = NnCardMeta {
            name: "x".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/somebody-else".into(),
                device: None,
                dtype: None,
                lora: None,
                custom: None,
            }),
        };
        let err = NnModelCard::new(id, meta).expect_err("mismatch must be rejected");
        assert!(err.contains("does not match card_id"), "got: {err}");
    }

    #[test]
    fn new_rejects_unknown_training_path_and_architecture() {
        let id = CardId::mint("x");
        let mut meta = NnCardMeta {
            name: "x".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "freestyle".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: None,
        };
        assert!(NnModelCard::new(id.clone(), meta.clone()).is_err());
        meta.training_path = "full_ft".into();
        meta.architecture = "gpt7".into();
        assert!(NnModelCard::new(id, meta).is_err());
    }

    #[test]
    fn from_merge_overrides_name_and_keeps_invariant() {
        let id = CardId::mint("merged");
        let meta = NnCardMeta {
            name: "file-stem-default".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "merged".into(),
            lineage: NnLineage {
                parent: Some("lora-card-1".into()),
                ..NnLineage::default()
            },
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: id.bundle_ref(),
                device: None,
                dtype: None,
                lora: None,
                custom: None,
            }),
        };
        let card = NnModelCard::from_merge(id, "user-name".into(), meta).expect("from_merge");
        assert_eq!(card.meta().name, "user-name");
        assert_eq!(card.meta().training_path, "merged");
    }

    // ─── channel axes: recorded, and checked against the bundle ──────

    /// A spec that declares `axis` and nothing else.
    fn spec_declaring(axis: ChannelAxis) -> Gpt2Custom {
        match axis {
            ChannelAxis::Conditioning => Gpt2Custom {
                cond_slots: Some(3),
                ..Gpt2Custom::default()
            },
            ChannelAxis::AllowedInput => Gpt2Custom {
                allowed_input: true,
                ..Gpt2Custom::default()
            },
        }
    }

    /// A custom-architecture card carrying `spec`.
    fn card_meta_with_spec(spec: Gpt2Custom) -> NnCardMeta {
        NnCardMeta {
            name: "channels".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-custom".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/channels".into(),
                device: None,
                dtype: None,
                lora: None,
                custom: Some(NnCustomBranch {
                    vocab: 32,
                    ctx: 8,
                    layers: 1,
                    heads: 2,
                    dim: 16,
                    spec,
                    moe: None,
                }),
            }),
        }
    }

    /// Both axes, in both directions, against a bundle that either
    /// carries the table or does not: the four quadrants the check is
    /// defined over.
    #[test]
    fn channel_verification_covers_every_quadrant() {
        for axis in ChannelAxis::ALL {
            let declared = card_meta_with_spec(spec_declaring(axis));
            let silent = card_meta_with_spec(Gpt2Custom::default());
            let with_table = ["wte.weight".to_string(), axis.tensor().to_string()];
            let without_table = ["wte.weight".to_string()];

            assert!(declared.declares_channel(axis));
            assert!(!silent.declares_channel(axis));

            // Declared + present: the pairing the run actually had.
            assert_eq!(declared.verify_channel_tensors(&with_table), Ok(()));
            // Declared + absent: the table the card claims is not there.
            assert_eq!(
                declared.verify_channel_tensors(&without_table),
                Err(ChannelMismatch::Missing { axis })
            );
            // Present + undeclared: the silent direction — a trained
            // channel the load path would never ask for.
            assert_eq!(
                silent.verify_channel_tensors(&with_table),
                Err(ChannelMismatch::Undeclared { axis })
            );
            // Neither: an ordinary card over an ordinary bundle.
            assert_eq!(silent.verify_channel_tensors(&without_table), Ok(()));
        }
    }

    /// The other pair a loader has to compare: the declaration against
    /// the config it is about to build from. A config that reads none
    /// of the channels — the reference architecture, or an
    /// architecture with no customization spec at all — drops every
    /// channel the Card declares, however complete the bundle is.
    #[test]
    fn a_config_that_reads_no_channel_is_refused_for_a_card_that_declares_one() {
        for axis in ChannelAxis::ALL {
            let declared = card_meta_with_spec(spec_declaring(axis));
            let silent = card_meta_with_spec(Gpt2Custom::default());

            // The config the Card was trained with reads it.
            assert_eq!(
                declared.verify_channels_consumed(Some(&spec_declaring(axis))),
                Ok(())
            );
            // A config rebuilt without the spec reads nothing.
            assert_eq!(
                declared.verify_channels_consumed(None),
                Err(ChannelMismatch::NotConsumed { axis })
            );
            // ... and so does one whose spec leaves the axis off.
            assert_eq!(
                declared.verify_channels_consumed(Some(&Gpt2Custom::default())),
                Err(ChannelMismatch::NotConsumed { axis })
            );
            // A Card that declares nothing has nothing to drop.
            assert_eq!(silent.verify_channels_consumed(None), Ok(()));
        }

        let message = ChannelMismatch::NotConsumed {
            axis: ChannelAxis::Conditioning,
        }
        .to_string();
        assert!(message.contains("cond_slots"), "got: {message}");
    }

    /// The message has to name both the tensor and the spec key, since
    /// the two ends of the disagreement are edited in different places.
    #[test]
    fn mismatch_messages_name_the_tensor_and_the_spec_field() {
        let missing = ChannelMismatch::Missing {
            axis: ChannelAxis::Conditioning,
        }
        .to_string();
        assert!(missing.contains("cond_slots"), "got: {missing}");
        assert!(
            missing.contains(crate::arch::COND_TABLE_TENSOR),
            "got: {missing}"
        );

        let undeclared = ChannelMismatch::Undeclared {
            axis: ChannelAxis::AllowedInput,
        }
        .to_string();
        assert!(undeclared.contains("allowed_input"), "got: {undeclared}");
        assert!(
            undeclared.contains(crate::arch::ALLOWED_TABLE_TENSOR),
            "got: {undeclared}"
        );
    }

    /// A reference-architecture card has no `custom` branch at all, so
    /// it declares neither channel — and a bundle that carries one is
    /// the same silence, reached from the other side.
    #[test]
    fn a_card_without_a_custom_branch_declares_no_channel() {
        let meta = NnCardMeta {
            name: "reference".into(),
            backend: "candle".into(),
            task: None,
            architecture: "gpt2-medium".into(),
            training_path: "full_ft".into(),
            lineage: NnLineage::default(),
            hyperparams: empty_object(),
            metrics: empty_object(),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/reference".into(),
                device: None,
                dtype: None,
                lora: None,
                custom: None,
            }),
        };
        assert!(!meta.declares_channel(ChannelAxis::Conditioning));
        assert!(!meta.declares_channel(ChannelAxis::AllowedInput));
        assert_eq!(meta.verify_channel_tensors(&["wte.weight"]), Ok(()));
        assert_eq!(
            meta.verify_channel_tensors(&["wte.weight", crate::arch::COND_TABLE_TENSOR]),
            Err(ChannelMismatch::Undeclared {
                axis: ChannelAxis::Conditioning
            })
        );
    }

    /// A card written before the channel axes existed has neither key
    /// in its spec table. It must still deserialize — and it reads as
    /// declaring nothing, which is what such a run was.
    #[test]
    fn deserialize_backwards_compat_spec_without_channel_axes() {
        let legacy = serde_json::json!({
            "name": "legacy-custom",
            "backend": "candle",
            "architecture": "gpt2-custom",
            "training_path": "full_ft",
            "candle": {
                "bundle_ref": "nn/legacy-custom",
                "custom": {
                    "vocab": 32,
                    "ctx": 8,
                    "layers": 1,
                    "heads": 2,
                    "dim": 16,
                    "spec": { "norm": "rmsnorm" }
                }
            }
        });
        let meta: NnCardMeta = serde_json::from_value(legacy).expect("legacy card deserialize");
        let spec = &meta
            .candle
            .as_ref()
            .and_then(|c| c.custom.as_ref())
            .expect("custom branch")
            .spec;
        assert_eq!(spec.cond_slots, None);
        assert!(!spec.allowed_input);
        assert_eq!(meta.verify_channel_tensors(&["wte.weight"]), Ok(()));
    }

    /// The axes serialize under the keys the check reports, so a Card
    /// on disk can be read (and corrected) by hand.
    #[test]
    fn declared_channels_serialize_under_their_spec_keys() {
        let meta = card_meta_with_spec(spec_declaring(ChannelAxis::Conditioning));
        let json = serde_json::to_value(&meta).expect("serialize");
        let spec = json
            .pointer("/candle/custom/spec")
            .expect("spec sub-object present");
        assert_eq!(spec.get("cond_slots"), Some(&serde_json::json!(3)));
        // `allowed_input` is a plain bool, so it is written either way.
        assert_eq!(spec.get("allowed_input"), Some(&serde_json::json!(false)));

        let allowed = card_meta_with_spec(spec_declaring(ChannelAxis::AllowedInput));
        let json = serde_json::to_value(&allowed).expect("serialize");
        let spec = json
            .pointer("/candle/custom/spec")
            .expect("spec sub-object present");
        assert_eq!(spec.get("allowed_input"), Some(&serde_json::json!(true)));
        assert!(
            spec.get("cond_slots").is_none(),
            "an unset Option is omitted, got: {spec}"
        );
    }

    /// The config -> Card projection carries the channel axes through
    /// without the caller naming them, so a run that built the tables
    /// produces a card that declares them.
    #[test]
    fn training_from_a_channelled_config_records_the_axis() {
        for axis in ChannelAxis::ALL {
            let cfg = Gpt2Config {
                custom: Some(spec_declaring(axis)),
                ..Gpt2Config::tiny()
            };
            let branch = NnCustomBranch::from_gpt2_config(&cfg).expect("custom config projects");
            let card = NnModelCard::from_training(
                CardId::mint("channelled"),
                "channelled",
                "gpt2-custom".into(),
                TrainingPath::FullFt,
                &test_ckpt(),
                &FullFtConfig::default(),
                Some(branch),
                None,
                None,
            )
            .expect("from_training");

            assert!(
                card.meta().declares_channel(axis),
                "{axis} must survive the config -> card projection"
            );
            assert_eq!(
                card.meta()
                    .verify_channel_tensors(&["wte.weight", axis.tensor()]),
                Ok(())
            );
        }
    }

    /// The load-time entry point reads the tensor names off a real
    /// bundle header and reaches the same verdicts.
    #[test]
    fn verification_against_a_bundle_on_disk() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("channelled.safetensors");
        let device = candle_core::Device::Cpu;
        let mut tensors: std::collections::HashMap<String, candle_core::Tensor> =
            std::collections::HashMap::new();
        tensors.insert(
            "wte.weight".into(),
            candle_core::Tensor::zeros((4, 2), candle_core::DType::F32, &device).expect("wte"),
        );
        tensors.insert(
            crate::arch::COND_TABLE_TENSOR.into(),
            candle_core::Tensor::zeros((3, 2), candle_core::DType::F32, &device).expect("cond"),
        );
        candle_core::safetensors::save(&tensors, &path).expect("write bundle");

        let declared = card_meta_with_spec(spec_declaring(ChannelAxis::Conditioning));
        assert_eq!(declared.verify_channel_tensors_in_bundle(&path), Ok(()));

        let silent = card_meta_with_spec(Gpt2Custom::default());
        assert_eq!(
            silent.verify_channel_tensors_in_bundle(&path),
            Err(ChannelMismatch::Undeclared {
                axis: ChannelAxis::Conditioning
            })
        );

        let allowed = card_meta_with_spec(spec_declaring(ChannelAxis::AllowedInput));
        assert_eq!(
            allowed.verify_channel_tensors_in_bundle(&path),
            Err(ChannelMismatch::Undeclared {
                axis: ChannelAxis::Conditioning
            }),
            "the first disagreeing axis is reported, in ALL order"
        );
    }

    /// A bundle that cannot be read is reported as unread, not as
    /// nothing to report.
    #[test]
    fn an_unreadable_bundle_is_refused_rather_than_passed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("absent.safetensors");
        let meta = card_meta_with_spec(Gpt2Custom::default());
        let err = meta
            .verify_channel_tensors_in_bundle(&missing)
            .expect_err("a missing bundle cannot be verified");
        match err {
            ChannelMismatch::BundleUnreadable { path, message } => {
                assert!(path.ends_with("absent.safetensors"), "got: {path}");
                assert!(!message.is_empty(), "the reader's reason must be carried");
            }
            other => panic!("expected BundleUnreadable, got {other:?}"),
        }
    }
}
