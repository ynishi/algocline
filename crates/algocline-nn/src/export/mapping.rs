//! A Card's fields in the vocabulary other tools read.
//!
//! One Card, four readers, one table:
//!
//! | Card | README YAML | GGUF | safetensors `__metadata__` |
//! |---|---|---|---|
//! | `training_path` `full_ft` / `lora` / `merged` | `base_model_relation` `finetune` / `adapter` / `merge` | — | `alc.training_path` |
//! | `training_path` `distillation` | not written; stated in the body | — | `alc.training_path` |
//! | `lineage.parent`, a Hub repo id | `base_model` | `general.base_model.*` | `alc.lineage.parent` |
//! | `lineage.parent`, anything else | body only | — | `alc.lineage.parent` |
//! | `lineage.training_data`, a Hub dataset id | `datasets` | `general.dataset.*` | `alc.lineage.training_data` |
//! | `lineage.teacher` / `.tokenizer` | body | — | `alc.lineage.teacher` / `.tokenizer` |
//! | `name` | body title | `general.name` | — |
//! | `architecture` | body | `general.architecture` (the writer's own) | `alc.architecture` |
//! | `hyperparams` / `metrics` | body | — | `alc.hyperparams` / `alc.metrics` (JSON strings) |
//! | card id | body | — | `alc.card_id` |
//! | the export's `license` | `license` | `general.license` | — |
//! | fixed | `tags: [algocline, candle]` | `general.tags` | `format = "pt"`, `alc.schema`, `alc.kind = "export"`, `alc.producer` |
//! | computed | body: the definition | — | `alc.tensor_sha256` |
//!
//! Never written: `library_name` (candle is not a registered Hub
//! library), `pipeline_tag` (a Card's `task` is free-form, not the Hub's
//! task vocabulary), `model-index`, `general.uuid`, and `model_type` /
//! `architectures` in `config.json` (the export does not claim
//! `transformers` compatibility).

use std::collections::BTreeMap;

use candle_core::quantized::gguf_file;
use serde_json::{Map, Value};

use super::bundle::TENSOR_SHA256_KEY;
use crate::card::{NnCardMeta, SUPPORTED_ARCHITECTURE_FAMILIES};
use crate::train::ckpt::{
    ARCHITECTURE_KEY, BUNDLE_SCHEMA, CARD_ID_KEY, FORMAT_KEY, FORMAT_PT, KIND_KEY, PRODUCER_KEY,
    SCHEMA_KEY,
};

/// Value of [`KIND_KEY`] in an exported bundle's header — the key set
/// [`CardExport::safetensors_metadata`] writes, as opposed to a trainer
/// checkpoint's.
pub const KIND_EXPORT: &str = "export";

/// Tags every export carries, in the README and in GGUF alike.
pub const TAGS: [&str; 2] = ["algocline", "candle"];

/// Longest Hub repo id, `org/name` together.
const HUB_ID_MAX_LEN: usize = 96;

/// Whether `s` names a Hugging Face Hub repository.
///
/// The rule: exactly `org/name`, at most 96 characters in all, where
/// each of the two segments
///
/// - is non-empty and every character is in `[A-Za-z0-9._-]` (so no
///   scheme, no whitespace, no second `/`),
/// - neither starts nor ends with `-` or `.`,
/// - contains neither `--` nor `..`;
///
/// and `s` is not one of algocline's own reference forms. A Card id has
/// no `/` at all (its alphabet is `[A-Za-z0-9_.-]`, see
/// [`crate::card::CardId`]), so it never passes; `cards/<id>` (the Card
/// reference form the lineage fields document) and `nn/<stem>` (a
/// bundle reference, [`crate::card::bundle_ref_for`]) have the Hub's
/// shape and are excluded by prefix.
///
/// This is a claim about form, not existence: nothing here asks the Hub
/// whether the repository is there. A local relative path of the same
/// form (`data/train.jsonl`) cannot be told apart from a Hub id and
/// passes.
pub fn is_hub_repo_id(s: &str) -> bool {
    if s.starts_with("cards/") || s.starts_with("nn/") || s.chars().count() > HUB_ID_MAX_LEN {
        return false;
    }
    let Some((org, name)) = s.split_once('/') else {
        return false;
    };
    let segment_ok = |seg: &str| {
        !seg.is_empty()
            && seg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            && !seg.starts_with(['-', '.'])
            && !seg.ends_with(['-', '.'])
            && !seg.contains("--")
            && !seg.contains("..")
    };
    segment_ok(org) && segment_ok(name)
}

/// The Hub's `base_model_relation` for a Card's `training_path`.
///
/// `None` for `distillation`: the Hub's four values (`adapter` /
/// `merge` / `quantized` / `finetune`) have none for it, and writing the
/// nearest one would state something the run did not do. `None` too for
/// a value this version does not know.
pub fn base_model_relation(training_path: &str) -> Option<&'static str> {
    match training_path {
        "full_ft" => Some("finetune"),
        "lora" => Some("adapter"),
        "merged" => Some("merge"),
        _ => None,
    }
}

/// The architecture family `arch` belongs to, by the rule
/// [`crate::card::validate_architecture`] uses: the bare family name, or
/// the family followed by `-`.
fn family_of(arch: &str) -> Option<&'static str> {
    SUPPORTED_ARCHITECTURE_FAMILIES.iter().copied().find(|f| {
        arch == *f
            || arch
                .strip_prefix(f)
                .is_some_and(|rest| rest.starts_with('-'))
    })
}

/// `v` rebuilt with every object's keys inserted in sorted order.
///
/// `serde_json`'s map is a `BTreeMap` by default and an insertion-order
/// map when something in the build turns on `preserve_order`. Inserting
/// in sorted order makes both print the same, so what these functions
/// write does not depend on the feature set of an unrelated crate.
fn canonical(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), canonical(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// Length of the longest run of backticks in `s`.
fn longest_backtick_run(s: &str) -> usize {
    let mut longest = 0;
    let mut run = 0;
    for c in s.chars() {
        if c == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    longest
}

/// `s` as Markdown code that nothing inside it can close.
///
/// Inline, the fence is one backtick longer than the longest run of
/// backticks in `s` (CommonMark closes a code span only on a run of the
/// same length), padded with a space when `s` holds a backtick so one at
/// either end does not merge into the fence. A value with a line break
/// cannot be an inline span at all — the break would end the list item
/// it sits in — so it becomes a fenced block, again with a fence longer
/// than any run inside.
fn code(s: &str) -> String {
    let longest = longest_backtick_run(s);
    if s.contains(['\n', '\r']) {
        let fence = "`".repeat((longest + 1).max(3));
        return format!("\n\n{fence}\n{s}\n{fence}\n");
    }
    let fence = "`".repeat(longest + 1);
    if longest > 0 {
        format!("{fence} {s} {fence}")
    } else {
        format!("{fence}{s}{fence}")
    }
}

/// `s` on one line: every run of `\n` / `\r` becomes a single space, so
/// a Card name cannot end the Markdown heading it is the title of.
fn one_line(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_break = false;
    for c in s.chars() {
        if c == '\n' || c == '\r' {
            if !in_break {
                out.push(' ');
            }
            in_break = true;
        } else {
            out.push(c);
            in_break = false;
        }
    }
    out
}

/// Split a Hub id into `(organization, name)`. Only called on ids that
/// passed [`is_hub_repo_id`].
fn split_hub_id(id: &str) -> Option<(&str, &str)> {
    id.split_once('/')
}

/// One Card, ready to be written out in the ecosystem's vocabulary.
///
/// Pure: everything here is a function of these four fields, so the
/// outputs can be checked without a store, a bundle or a filesystem.
#[derive(Debug, Clone, Copy)]
pub struct CardExport<'a> {
    /// The Card's id.
    pub card_id: &'a str,
    /// The Card's `[metadata.nn]` block.
    pub meta: &'a NnCardMeta,
    /// SPDX license id, when the caller gave one. The Card has no
    /// license field, so this is the only source.
    pub license: Option<&'a str>,
    /// Program that wrote the export (`algocline <version>`,
    /// [`crate::train::producer`]).
    pub producer: &'a str,
}

impl CardExport<'_> {
    /// `lineage.parent`, when it is a Hub repo id.
    fn hub_parent(&self) -> Option<&str> {
        self.meta
            .lineage
            .parent
            .as_deref()
            .filter(|p| is_hub_repo_id(p))
    }

    /// `lineage.training_data`, when it is a Hub dataset id.
    fn hub_dataset(&self) -> Option<&str> {
        self.meta
            .lineage
            .training_data
            .as_deref()
            .filter(|d| is_hub_repo_id(d))
    }

    /// `config.json`: the shape algocline's loader rebuilds the model
    /// from.
    ///
    /// Built from the preset for a named variant, and from
    /// `candle.custom` for a custom GPT-2 — through the same
    /// [`NnCardMeta::gpt2_config`] / [`NnCardMeta::tinyllama_config`]
    /// the engine's `load_handle` uses, so a Card the loader refuses is
    /// refused here with the same message. No `model_type` /
    /// `architectures`: this does not claim `transformers`
    /// compatibility.
    ///
    /// Pretty-printed with its keys sorted, so the same Card gives the
    /// same bytes.
    ///
    /// # Errors
    ///
    /// The loader's refusals — an unknown variant with no shape block
    /// (a custom architecture without `candle.custom`), custom+MoE, a
    /// declared channel the rebuilt config would not read — and a
    /// family with no loader (`llama`, `qwen2`, `phi`, `gemma`), for
    /// which there is no shape this crate can rebuild to describe.
    pub fn config_json(&self) -> Result<String, String> {
        let meta = self.meta;
        let family = family_of(&meta.architecture).ok_or_else(|| {
            format!(
                "architecture {:?} is not a known family (expected one of \
                 {SUPPORTED_ARCHITECTURE_FAMILIES:?})",
                meta.architecture
            )
        })?;
        let mut obj = Map::new();
        obj.insert("alc_schema".into(), Value::from(BUNDLE_SCHEMA));
        obj.insert(
            "architecture".into(),
            Value::from(meta.architecture.clone()),
        );
        obj.insert("family".into(), Value::from(family));
        match family {
            "gpt2" => {
                let cfg = meta.gpt2_config()?;
                obj.insert("vocab".into(), Value::from(cfg.vocab));
                obj.insert("ctx".into(), Value::from(cfg.ctx));
                obj.insert("layers".into(), Value::from(cfg.layers));
                obj.insert("heads".into(), Value::from(cfg.heads));
                obj.insert("kv_heads".into(), Value::from(cfg.effective_kv_heads()));
                obj.insert("dim".into(), Value::from(cfg.dim));
                obj.insert("eps".into(), Value::from(cfg.eps));
                if let Some(spec) = cfg.custom.as_ref() {
                    let spec = serde_json::to_value(spec)
                        .map_err(|e| format!("serialise candle.custom.spec: {e}"))?;
                    obj.insert("spec".into(), spec);
                }
            }
            "tinyllama" => {
                let cfg = meta.tinyllama_config()?;
                obj.insert("vocab".into(), Value::from(cfg.vocab));
                obj.insert("ctx".into(), Value::from(cfg.ctx));
                obj.insert("layers".into(), Value::from(cfg.layers));
                obj.insert("heads".into(), Value::from(cfg.heads));
                obj.insert("kv_heads".into(), Value::from(cfg.kv_heads));
                obj.insert("dim".into(), Value::from(cfg.dim));
                obj.insert("hidden_dim".into(), Value::from(cfg.hidden_dim));
                obj.insert("rope_theta".into(), Value::from(cfg.rope_theta));
                obj.insert("eps".into(), Value::from(cfg.eps));
            }
            other => {
                return Err(format!(
                    "architecture {:?} is family `{other}`, which has no card loader; \
                     config.json describes a shape algocline can rebuild, and there is none \
                     to describe",
                    meta.architecture
                ))
            }
        }
        serde_json::to_string_pretty(&canonical(&Value::Object(obj)))
            .map_err(|e| format!("serialise config.json: {e}"))
    }

    /// The safetensors `__metadata__` map, without `alc.tensor_sha256`
    /// (which [`super::rewrite_with_metadata`] computes and adds).
    ///
    /// `format = "pt"` is the one key readers share; everything else is
    /// algocline's and sits under `alc.`. `alc.hyperparams` /
    /// `alc.metrics` are compact JSON with sorted keys, `alc.config` is
    /// the [`Self::config_json`] string, and the `alc.lineage.*` keys
    /// are present only for the fields the Card sets.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::config_json`] refuses.
    pub fn safetensors_metadata(&self) -> Result<BTreeMap<String, String>, String> {
        let meta = self.meta;
        let mut out = BTreeMap::new();
        out.insert(FORMAT_KEY.to_string(), FORMAT_PT.to_string());
        out.insert(SCHEMA_KEY.to_string(), BUNDLE_SCHEMA.to_string());
        out.insert(KIND_KEY.to_string(), KIND_EXPORT.to_string());
        out.insert(PRODUCER_KEY.to_string(), self.producer.to_string());
        out.insert(CARD_ID_KEY.to_string(), self.card_id.to_string());
        out.insert(ARCHITECTURE_KEY.to_string(), meta.architecture.clone());
        out.insert("alc.training_path".into(), meta.training_path.clone());
        let lineage = [
            ("alc.lineage.parent", &meta.lineage.parent),
            ("alc.lineage.teacher", &meta.lineage.teacher),
            ("alc.lineage.training_data", &meta.lineage.training_data),
            ("alc.lineage.tokenizer", &meta.lineage.tokenizer),
        ];
        for (key, value) in lineage {
            if let Some(v) = value {
                out.insert(key.to_string(), v.clone());
            }
        }
        out.insert(
            "alc.hyperparams".into(),
            serde_json::to_string(&canonical(&meta.hyperparams))
                .map_err(|e| format!("serialise hyperparams: {e}"))?,
        );
        out.insert(
            "alc.metrics".into(),
            serde_json::to_string(&canonical(&meta.metrics))
                .map_err(|e| format!("serialise metrics: {e}"))?,
        );
        out.insert("alc.config".into(), self.config_json()?);
        Ok(out)
    }

    /// The model card: YAML front matter the Hub reads, then a Markdown
    /// body with everything the front matter has no key for.
    ///
    /// Front matter: `tags`; `license` only when one was given;
    /// `base_model` only when `lineage.parent` is a Hub repo id
    /// ([`is_hub_repo_id`]); `base_model_relation` only when
    /// `base_model` is written and the training path has a Hub value
    /// ([`base_model_relation`]); `datasets` only when
    /// `lineage.training_data` is a Hub dataset id.
    ///
    /// `tensor_sha256` is the digest the bundle was written with; the
    /// body states it and what it covers.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::safetensors_metadata`] refuses, since the body
    /// lists that map's keys.
    pub fn readme(&self, tensor_sha256: &str) -> Result<String, String> {
        let meta = self.meta;
        let hub_parent = self.hub_parent();
        let hub_dataset = self.hub_dataset();
        let relation = hub_parent.and(base_model_relation(&meta.training_path));

        let mut y = String::from("---\n");
        y.push_str(&format!("tags: [{}]\n", TAGS.join(", ")));
        if let Some(license) = self.license {
            // Quoted: a JSON string literal is a valid YAML
            // double-quoted scalar, and the value is the caller's.
            let quoted = serde_json::to_string(license)
                .map_err(|e| format!("encode license {license:?}: {e}"))?;
            y.push_str(&format!("license: {quoted}\n"));
        }
        if let Some(parent) = hub_parent {
            y.push_str(&format!("base_model: {parent}\n"));
            if let Some(rel) = relation {
                y.push_str(&format!("base_model_relation: {rel}\n"));
            }
        }
        if let Some(data) = hub_dataset {
            y.push_str(&format!("datasets:\n- {data}\n"));
        }
        y.push_str("---\n");

        let mut b = String::new();
        b.push_str(&format!("\n# {}\n\n", one_line(&meta.name)));
        b.push_str(&format!(
            "Exported from algocline ({}) by `alc.nn.card.export`.\n\n",
            code(self.producer)
        ));
        b.push_str(&format!("- Card id: {}\n", code(self.card_id)));
        b.push_str(&format!("- Architecture: {}\n", code(&meta.architecture)));
        b.push_str(&format!("- Training path: {}\n", code(&meta.training_path)));
        if let Some(task) = meta.task.as_deref() {
            b.push_str(&format!("- Task: {}\n", code(task)));
        }
        b.push('\n');
        b.push_str(&training_path_paragraph(
            &meta.training_path,
            hub_parent,
            relation,
        ));

        b.push_str("\n## Lineage\n\n");
        b.push_str(&lineage_line(
            "Parent",
            meta.lineage.parent.as_deref(),
            "a Hugging Face model repository, written as `base_model` above",
        ));
        b.push_str(&lineage_line(
            "Teacher",
            meta.lineage.teacher.as_deref(),
            "",
        ));
        b.push_str(&lineage_line(
            "Training data",
            meta.lineage.training_data.as_deref(),
            "a Hugging Face dataset repository, written as `datasets` above",
        ));
        b.push_str(&lineage_line(
            "Tokenizer",
            meta.lineage.tokenizer.as_deref(),
            "",
        ));

        b.push_str("\n## Hyperparameters\n\n");
        b.push_str(&json_block(&meta.hyperparams)?);
        b.push_str("\n## Metrics\n\n");
        b.push_str(&json_block(&meta.metrics)?);

        b.push_str("\n## Files\n\n");
        b.push_str(
            "- `model.safetensors` — the weights, with the keys below in its `__metadata__` \
             header.\n",
        );
        b.push_str(
            "- `config.json` — the shape algocline's loader rebuilds the model from. It names \
             no `model_type` or `architectures`: this export does not claim `transformers` \
             compatibility.\n",
        );
        b.push_str("- `model.gguf` — present only when the export was asked for one.\n");

        let mut keys: Vec<String> = self.safetensors_metadata()?.into_keys().collect();
        keys.push(TENSOR_SHA256_KEY.to_string());
        keys.sort();
        b.push_str("\n## safetensors `__metadata__`\n\n");
        let listed: Vec<String> = keys.iter().map(|k| code(k)).collect();
        b.push_str(&format!("Keys: {}.\n\n", listed.join(", ")));
        b.push_str(
            "`format` is `pt`, the one key safetensors readers share. Every other key is \
             algocline's and sits under `alc.`.\n",
        );

        b.push_str(&format!("\n## `{TENSOR_SHA256_KEY}`\n\n"));
        b.push_str(&format!("{}\n\n", code(tensor_sha256)));
        b.push_str(
            "The lowercase hex SHA-256 of the data section of `model.safetensors` — every byte \
             after the header. That is the range ModelSpec's `hash_sha256` covers, written \
             without its `0x` prefix. It is not the Hugging Face Hub's LFS `sha256`, which \
             covers the whole file, header included.\n",
        );

        b.push_str("\n## License\n\n");
        match self.license {
            Some(license) => b.push_str(&format!(
                "{}, as given at export. The Card itself records no license.\n",
                code(license)
            )),
            None => b.push_str(
                "No license was given at export, so the front matter sets none. The Card itself \
                 records no license.\n",
            ),
        }

        Ok(format!("{y}{b}"))
    }

    /// The `general.*` keys a GGUF export adds on top of the
    /// architecture's own.
    ///
    /// `general.name` is the Card's name (it replaces the preset variant
    /// the writer puts there), `general.license` when one was given,
    /// `general.tags`, and — when the lineage names Hub repositories —
    /// `general.base_model.*` and `general.dataset.*` in the form
    /// llama.cpp's converter writes from a model card: a `count`, then
    /// `name` / `organization` / `repo_url` for entry `0`.
    pub fn gguf_metadata(&self) -> Vec<(String, gguf_file::Value)> {
        use gguf_file::Value as G;
        let mut out = vec![(
            "general.name".to_string(),
            G::String(self.meta.name.clone()),
        )];
        if let Some(license) = self.license {
            out.push(("general.license".into(), G::String(license.to_string())));
        }
        out.push((
            "general.tags".into(),
            G::Array(TAGS.iter().map(|t| G::String((*t).to_string())).collect()),
        ));
        if let Some((org, name)) = self.hub_parent().and_then(split_hub_id) {
            out.push(("general.base_model.count".into(), G::U32(1)));
            out.push(("general.base_model.0.name".into(), G::String(name.into())));
            out.push((
                "general.base_model.0.organization".into(),
                G::String(org.into()),
            ));
            out.push((
                "general.base_model.0.repo_url".into(),
                G::String(format!("https://huggingface.co/{org}/{name}")),
            ));
        }
        if let Some((org, name)) = self.hub_dataset().and_then(split_hub_id) {
            out.push(("general.dataset.count".into(), G::U32(1)));
            out.push(("general.dataset.0.name".into(), G::String(name.into())));
            out.push((
                "general.dataset.0.organization".into(),
                G::String(org.into()),
            ));
            out.push((
                "general.dataset.0.repo_url".into(),
                G::String(format!("https://huggingface.co/datasets/{org}/{name}")),
            ));
        }
        out
    }
}

/// What the body says about the training path.
fn training_path_paragraph(
    training_path: &str,
    hub_parent: Option<&str>,
    relation: Option<&str>,
) -> String {
    let what = match training_path {
        "full_ft" => "Trained by full fine-tuning.".to_string(),
        "lora" => "A LoRA adapter.".to_string(),
        "merged" => "Base weights with a LoRA adapter merged into them.".to_string(),
        "distillation" => "Trained by distillation. The Hub's `base_model_relation` has four \
             values — `adapter`, `merge`, `quantized`, `finetune` — and none of them is \
             distillation, so the front matter sets none. When the relation is absent the Hub \
             infers one, and this model may be shown as a fine-tune; nothing in the file can \
             prevent that."
            .to_string(),
        other => format!("Training path {}.", code(other)),
    };
    match (hub_parent, relation) {
        (Some(parent), Some(rel)) => format!(
            "{what} The front matter records it as `base_model_relation: {rel}` of {}.\n",
            code(parent)
        ),
        _ => format!("{what}\n"),
    }
}

/// One lineage bullet. `hub_note` is appended when the value is a Hub
/// id and a note was given.
fn lineage_line(label: &str, value: Option<&str>, hub_note: &str) -> String {
    match value {
        None => format!("- {label}: none recorded\n"),
        Some(v) if !hub_note.is_empty() && is_hub_repo_id(v) => {
            format!("- {label}: {} — {hub_note}\n", code(v))
        }
        Some(v) if !hub_note.is_empty() => format!(
            "- {label}: {} — an algocline reference, not a Hub repository, so the front matter \
             does not name it\n",
            code(v)
        ),
        Some(v) => format!("- {label}: {}\n", code(v)),
    }
}

/// A fenced JSON block, keys sorted.
fn json_block(v: &Value) -> Result<String, String> {
    let text =
        serde_json::to_string_pretty(&canonical(v)).map_err(|e| format!("serialise: {e}"))?;
    // A string value may hold a run of backticks; the fence outlasts it.
    let fence = "`".repeat((longest_backtick_run(&text) + 1).max(3));
    Ok(format!("{fence}json\n{text}\n{fence}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{Activation, Gpt2Custom};
    use crate::card::{NnCandleBranch, NnCustomBranch, NnLineage, NnMoeBranch};

    fn meta(architecture: &str, training_path: &str) -> NnCardMeta {
        NnCardMeta {
            name: "demo model".into(),
            backend: "candle".into(),
            task: None,
            architecture: architecture.into(),
            training_path: training_path.into(),
            lineage: NnLineage::default(),
            hyperparams: serde_json::json!({ "steps": 3, "lr": 0.001 }),
            metrics: serde_json::json!({ "train_loss": 1.5 }),
            candle: Some(NnCandleBranch {
                bundle_ref: "nn/demo_1".into(),
                device: None,
                dtype: None,
                lora: None,
                custom: None,
            }),
        }
    }

    fn export<'a>(meta: &'a NnCardMeta, license: Option<&'a str>) -> CardExport<'a> {
        CardExport {
            card_id: "demo_1",
            meta,
            license,
            producer: "algocline 9.9.9",
        }
    }

    /// The front matter, between the two `---` lines.
    fn front_matter(readme: &str) -> &str {
        let rest = readme.strip_prefix("---\n").expect("opens with ---");
        let end = rest.find("\n---\n").expect("closes with ---");
        &rest[..end + 1]
    }

    fn with_parent(mut m: NnCardMeta, parent: &str) -> NnCardMeta {
        m.lineage.parent = Some(parent.into());
        m
    }

    #[test]
    fn the_hub_id_rule() {
        for ok in ["openai-community/gpt2", "org/name.v2", "a_b/c-d"] {
            assert!(is_hub_repo_id(ok), "{ok}");
        }
        for no in [
            "gpt2",                     // no `/`
            "demo_1_1726000000000000",  // a Card id
            "cards/haiku-run-042",      // algocline's Card reference form
            "nn/base-gpt2-medium",      // a bundle reference
            "a/b/c",                    // two `/`
            "/name",                    // empty org
            "org/",                     // empty name
            "https://huggingface.co/x", // a URL
            "org/na me",                // whitespace
            "hf:org/name",              // a scheme
            "../x",                     // a leading `.` (and `..`)
            "-a/b",                     // a leading `-`
            "a-/b",                     // a trailing `-`
            "a--b/c",                   // `--`
            "a./b",                     // a trailing `.`
            "a/b..c",                   // `..`
        ] {
            assert!(!is_hub_repo_id(no), "{no}");
        }
        // The length limit covers the whole id: 96 passes, 97 does not.
        let at_limit = format!("org/{}", "a".repeat(92));
        assert_eq!(at_limit.len(), 96);
        assert!(is_hub_repo_id(&at_limit));
        let past = format!("org/{}", "a".repeat(93));
        assert_eq!(past.len(), 97);
        assert!(!is_hub_repo_id(&past));
        // A local relative path of the same form is indistinguishable,
        // which is the documented limitation.
        assert!(is_hub_repo_id("data/train.jsonl"));
    }

    #[test]
    fn code_spans_cannot_be_closed_from_inside() {
        assert_eq!(code("plain"), "`plain`");
        assert_eq!(code("a`b"), "`` a`b ``");
        assert_eq!(code("x```y"), "```` x```y ````");
        let block = code("line1\nline2");
        assert!(block.contains("\n```\nline1\nline2\n```\n"), "{block:?}");
        let block = code("a\n````b");
        assert!(block.contains("\n`````\na\n````b\n`````\n"), "{block:?}");
    }

    #[test]
    fn a_card_name_with_line_breaks_stays_one_heading() {
        let mut m = meta("gpt2-tiny", "full_ft");
        m.name = "evil\n\r\n## injected\rtail".into();
        let readme = export(&m, None).readme(&"0".repeat(64)).unwrap();
        assert!(readme.contains("\n# evil ## injected tail\n"), "{readme}");
        assert!(!readme.contains("\n## injected"), "{readme}");
    }

    #[test]
    fn a_json_block_outlasts_backticks_in_its_values() {
        let mut m = meta("gpt2-tiny", "full_ft");
        m.hyperparams = serde_json::json!({ "note": "```" });
        let readme = export(&m, None).readme(&"0".repeat(64)).unwrap();
        assert!(readme.contains("````json\n"), "{readme}");
    }

    #[test]
    fn each_training_path_maps_to_its_relation_and_distillation_to_none() {
        let cases = [
            ("full_ft", Some("finetune")),
            ("lora", Some("adapter")),
            ("merged", Some("merge")),
            ("distillation", None),
        ];
        for (path, want) in cases {
            assert_eq!(base_model_relation(path), want, "{path}");
            let m = with_parent(meta("gpt2-tiny", path), "openai-community/gpt2");
            let readme = export(&m, None).readme(&"a".repeat(64)).unwrap();
            let fm = front_matter(&readme);
            assert!(fm.contains("base_model: openai-community/gpt2\n"), "{fm}");
            match want {
                Some(rel) => assert!(
                    fm.contains(&format!("base_model_relation: {rel}\n")),
                    "{path}: {fm}"
                ),
                None => assert!(!fm.contains("base_model_relation"), "{path}: {fm}"),
            }
        }
        let m = with_parent(meta("gpt2-tiny", "distillation"), "openai-community/gpt2");
        let readme = export(&m, None).readme(&"a".repeat(64)).unwrap();
        assert!(readme.contains("Trained by distillation"), "{readme}");
        assert!(readme.contains("none of them is"), "{readme}");
    }

    #[test]
    fn a_hub_parent_is_written_and_a_card_parent_is_not() {
        let hub = with_parent(meta("gpt2-tiny", "full_ft"), "openai-community/gpt2");
        let e = export(&hub, None);
        let fm_readme = e.readme(&"0".repeat(64)).unwrap();
        assert!(front_matter(&fm_readme).contains("base_model: openai-community/gpt2"));
        let gguf: BTreeMap<String, String> = e
            .gguf_metadata()
            .into_iter()
            .filter_map(|(k, v)| v.to_string().ok().map(|s| (k, s.clone())))
            .collect();
        assert_eq!(gguf["general.base_model.0.name"], "gpt2");
        assert_eq!(
            gguf["general.base_model.0.organization"],
            "openai-community"
        );
        assert_eq!(
            gguf["general.base_model.0.repo_url"],
            "https://huggingface.co/openai-community/gpt2"
        );
        let count = e
            .gguf_metadata()
            .into_iter()
            .find(|(k, _)| k == "general.base_model.count")
            .expect("count")
            .1;
        assert_eq!(count.to_u32().unwrap(), 1);

        let card = with_parent(meta("gpt2-tiny", "merged"), "cards/domain-lora-042");
        let e = export(&card, None);
        let readme = e.readme(&"0".repeat(64)).unwrap();
        let fm = front_matter(&readme);
        assert!(!fm.contains("base_model"), "{fm}");
        assert!(
            readme.contains("`cards/domain-lora-042` — an algocline reference"),
            "the body still names it: {readme}"
        );
        assert!(!e
            .gguf_metadata()
            .iter()
            .any(|(k, _)| k.starts_with("general.base_model")));
        // And the safetensors header keeps it either way.
        let st = e.safetensors_metadata().unwrap();
        assert_eq!(st["alc.lineage.parent"], "cards/domain-lora-042");
    }

    #[test]
    fn a_hub_dataset_is_written_as_datasets() {
        let mut m = meta("gpt2-tiny", "full_ft");
        m.lineage.training_data = Some("org/corpus".into());
        let e = export(&m, None);
        let readme = e.readme(&"0".repeat(64)).unwrap();
        assert!(front_matter(&readme).contains("datasets:\n- org/corpus\n"));
        let urls: Vec<String> = e
            .gguf_metadata()
            .into_iter()
            .filter(|(k, _)| k == "general.dataset.0.repo_url")
            .map(|(_, v)| v.to_string().unwrap().clone())
            .collect();
        assert_eq!(urls, vec!["https://huggingface.co/datasets/org/corpus"]);
    }

    #[test]
    fn license_present_and_absent() {
        let m = meta("gpt2-tiny", "full_ft");
        let with = export(&m, Some("apache-2.0"))
            .readme(&"0".repeat(64))
            .unwrap();
        assert!(front_matter(&with).contains("license: \"apache-2.0\"\n"));
        assert!(export(&m, Some("apache-2.0"))
            .gguf_metadata()
            .iter()
            .any(|(k, v)| k == "general.license"
                && v.to_string().map(|s| s == "apache-2.0").unwrap_or(false)));

        let without = export(&m, None).readme(&"0".repeat(64)).unwrap();
        assert!(!front_matter(&without).contains("license"));
        assert!(without.contains("No license was given at export"));
        assert!(!export(&m, None)
            .gguf_metadata()
            .iter()
            .any(|(k, _)| k == "general.license"));
    }

    #[test]
    fn no_forbidden_front_matter_keys() {
        let mut m = with_parent(meta("gpt2-tiny", "full_ft"), "openai-community/gpt2");
        m.task = Some("classification".into());
        m.lineage.training_data = Some("org/corpus".into());
        let readme = export(&m, Some("mit")).readme(&"0".repeat(64)).unwrap();
        let fm = front_matter(&readme);
        for key in ["library_name", "pipeline_tag", "model-index", "model_type"] {
            assert!(!fm.contains(key), "{key} in {fm}");
        }
        assert!(fm.starts_with("tags: [algocline, candle]\n"), "{fm}");
        // Every front-matter line is `key: value` or a list item.
        for line in fm.lines() {
            assert!(
                line.starts_with("- ") || line.split_once(": ").is_some() || line.ends_with(':'),
                "{line:?}"
            );
        }
    }

    #[test]
    fn the_safetensors_map_carries_the_card() {
        let mut m = meta("gpt2-tiny", "full_ft");
        m.lineage.tokenizer = Some("gpt2".into());
        let st = export(&m, Some("mit")).safetensors_metadata().unwrap();
        assert_eq!(st["format"], "pt");
        assert_eq!(st["alc.schema"], "1");
        assert_eq!(st["alc.kind"], "export");
        assert_eq!(st["alc.producer"], "algocline 9.9.9");
        assert_eq!(st["alc.card_id"], "demo_1");
        assert_eq!(st["alc.architecture"], "gpt2-tiny");
        assert_eq!(st["alc.training_path"], "full_ft");
        assert_eq!(st["alc.lineage.tokenizer"], "gpt2");
        assert!(!st.contains_key("alc.lineage.parent"));
        assert!(!st.contains_key(TENSOR_SHA256_KEY), "added by the rewrite");
        // Sorted, compact.
        assert_eq!(st["alc.hyperparams"], r#"{"lr":0.001,"steps":3}"#);
        assert_eq!(st["alc.metrics"], r#"{"train_loss":1.5}"#);
        assert_eq!(
            st["alc.config"],
            export(&m, None).config_json().unwrap(),
            "the config.json string"
        );
        // The license is the README's and GGUF's, not the header's.
        assert!(!st.keys().any(|k| k.contains("license")));
    }

    #[test]
    fn config_json_for_a_named_gpt2_preset() {
        let m = meta("gpt2-tiny", "full_ft");
        let text = export(&m, None).config_json().unwrap();
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["alc_schema"], "1");
        assert_eq!(v["architecture"], "gpt2-tiny");
        assert_eq!(v["family"], "gpt2");
        assert_eq!(v["vocab"], 64);
        assert_eq!(v["ctx"], 16);
        assert_eq!(v["layers"], 2);
        assert_eq!(v["heads"], 2);
        assert_eq!(v["dim"], 32);
        assert!(v.get("spec").is_none());
        assert!(v.get("model_type").is_none() && v.get("architectures").is_none());
    }

    #[test]
    fn config_json_for_a_custom_gpt2_carries_the_spec() {
        let mut m = meta("gpt2-custom", "full_ft");
        m.candle.as_mut().unwrap().custom = Some(NnCustomBranch {
            vocab: 96,
            ctx: 32,
            layers: 2,
            heads: 2,
            dim: 32,
            spec: Gpt2Custom {
                act: Activation::SwiGlu,
                kv_heads: Some(1),
                ..Gpt2Custom::default()
            },
            moe: None,
        });
        let v: Value = serde_json::from_str(&export(&m, None).config_json().unwrap()).unwrap();
        assert_eq!(v["vocab"], 96);
        assert_eq!(v["kv_heads"], 1);
        assert_eq!(v["spec"]["act"], "swiglu");
    }

    #[test]
    fn config_json_for_tinyllama() {
        let m = meta("tinyllama-tiny", "full_ft");
        let v: Value = serde_json::from_str(&export(&m, None).config_json().unwrap()).unwrap();
        assert_eq!(v["family"], "tinyllama");
        assert_eq!(v["kv_heads"], 1);
        assert_eq!(v["hidden_dim"], 128);
    }

    #[test]
    fn config_json_refuses_what_the_loader_refuses() {
        // A custom architecture with no shape block.
        let m = meta("gpt2-custom", "full_ft");
        let err = export(&m, None).config_json().unwrap_err();
        assert!(err.contains("metadata.nn.candle.custom is absent"), "{err}");
        assert_eq!(err, m.gpt2_config().unwrap_err(), "one rule, one message");

        // Custom + MoE.
        let mut moe = meta("gpt2-custom", "full_ft");
        moe.candle.as_mut().unwrap().custom = Some(NnCustomBranch {
            vocab: 64,
            ctx: 16,
            layers: 2,
            heads: 2,
            dim: 32,
            spec: Gpt2Custom::default(),
            moe: Some(NnMoeBranch {
                n_experts: 2,
                top_k: 1,
                alpha: 0.01,
            }),
        });
        let err = export(&moe, None).config_json().unwrap_err();
        assert!(err.contains("custom+MoE"), "{err}");

        // A family with no loader.
        let llama = meta("llama-3.2-1b", "full_ft");
        let err = export(&llama, None).config_json().unwrap_err();
        assert!(err.contains("no card loader"), "{err}");
        // And the refusal reaches the header map too.
        assert!(export(&llama, None).safetensors_metadata().is_err());
    }

    #[test]
    fn the_outputs_are_a_function_of_the_card() {
        let m = with_parent(meta("gpt2-tiny", "full_ft"), "openai-community/gpt2");
        let a = export(&m, Some("mit"));
        assert_eq!(a.readme("x").unwrap(), a.readme("x").unwrap());
        assert_eq!(a.config_json().unwrap(), a.config_json().unwrap());
        assert_eq!(
            a.safetensors_metadata().unwrap(),
            a.safetensors_metadata().unwrap()
        );
    }

    #[test]
    fn the_readme_body_states_the_card() {
        let mut m = with_parent(meta("gpt2-tiny", "full_ft"), "openai-community/gpt2");
        m.task = Some("classification".into());
        m.lineage.teacher = Some("cards/teacher-1".into());
        let hash = "b".repeat(64);
        let readme = export(&m, None).readme(&hash).unwrap();
        assert!(readme.contains("\n# demo model\n"), "{readme}");
        assert!(readme.contains("Card id: `demo_1`"));
        assert!(readme.contains("Architecture: `gpt2-tiny`"));
        assert!(readme.contains("Task: `classification`"));
        assert!(readme.contains("Teacher: `cards/teacher-1`"));
        assert!(readme.contains("Tokenizer: none recorded"));
        assert!(readme.contains("\"steps\": 3"));
        assert!(readme.contains("\"train_loss\": 1.5"));
        assert!(readme.contains(&format!("`{hash}`")));
        assert!(readme.contains("`alc.tensor_sha256`"));
        assert!(readme.contains("not the Hugging Face Hub's LFS `sha256`"));
        assert!(readme.contains("`alc.config`"), "the key list: {readme}");
    }
}
