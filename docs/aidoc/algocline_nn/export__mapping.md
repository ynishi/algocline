# algocline-nn::export::mapping

A Card's fields in the vocabulary other tools read.

One Card, four readers, one table:

| Card | README YAML | GGUF | safetensors `__metadata__` |
|---|---|---|---|
| `training_path` `full_ft` / `lora` / `merged` | `base_model_relation` `finetune` / `adapter` / `merge` | — | `alc.training_path` |
| `training_path` `distillation` | not written; stated in the body | — | `alc.training_path` |
| `lineage.parent`, a Hub repo id | `base_model` | `general.base_model.*` | `alc.lineage.parent` |
| `lineage.parent`, anything else | body only | — | `alc.lineage.parent` |
| `lineage.training_data`, a Hub dataset id | `datasets` | `general.dataset.*` | `alc.lineage.training_data` |
| `lineage.teacher` / `.tokenizer` | body | — | `alc.lineage.teacher` / `.tokenizer` |
| `name` | body title | `general.name` | — |
| `architecture` | body | `general.architecture` (the writer's own) | `alc.architecture` |
| `hyperparams` / `metrics` | body | — | `alc.hyperparams` / `alc.metrics` (JSON strings) |
| card id | body | — | `alc.card_id` |
| the export's `license` | `license` | `general.license` | — |
| fixed | `tags: [algocline, candle]` | `general.tags` | `format = "pt"`, `alc.schema`, `alc.kind = "export"`, `alc.producer` |
| computed | body: the definition | — | `alc.tensor_sha256` |

Never written: `library_name` (candle is not a registered Hub
library), `pipeline_tag` (a Card's `task` is free-form, not the Hub's
task vocabulary), `model-index`, `general.uuid`, and `model_type` /
`architectures` in `config.json` (the export does not claim
`transformers` compatibility).

## Functions

- `base_model_relation` — The Hub's `base_model_relation` for a Card's `training_path`.
- `is_hub_repo_id` — Whether `s` names a Hugging Face Hub repository.

## Types

- `CardExport` — One Card, ready to be written out in the ecosystem's vocabulary.

## Constants

- `KIND_EXPORT` — Value of [`KIND_KEY`] in an exported bundle's header — the key set
- `TAGS` — Tags every export carries, in the README and in GGUF alike.

