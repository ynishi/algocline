# algocline-nn::card

Card metadata schema for `alc.nn.card.*`.

Mirrors the `[metadata.nn]` TOML block written by the engine bridge
(`bridge/nn_card.rs`) when it assembles the Card create payload.
Downstream training paths (Full FT / LoRA / Distillation) populate
`hyperparams` / `metrics` / `lineage` uniformly through this schema.

`hyperparams` and `metrics` are free-form JSON pass-through so trainer
subtasks can extend without reshaping this crate.

The Card foundation leaves `NnCandleBranch::lora` as `None`. A
later LoRA follow-up populates it via the [`NnLoraBranch`]
sub-struct without breaking foundation serialization
(`skip_serializing_if = "Option::is_none"`).

## Functions

- `bundle_ref_for` — Logical `"nn/<stem>"` bundle reference.
- `sanitize_stem` — Collapse a free-form name into the `[A-Za-z0-9_-]` stem alphabet
- `unique_stem` — Mint a unique, filesystem-safe stem from a free-form `name`.
- `validate_architecture` — Validate that `arch` starts with a known family prefix from
- `validate_training_path` — Validate that `training_path` is one of

## Types

- `CardId` — Validated identifier of an `nn_model` Card.
- `ChannelAxis` — An optional input channel a custom-architecture run can be trained
- `ChannelMismatch` — A Card's declared channels and the tensors in its bundle disagree.
- `NnCandleBranch` — Content of `[metadata.nn.candle]`.
- `NnCardMeta` — Content of `[metadata.nn]`.
- `NnCustomBranch` — Content of `[metadata.nn.candle.custom]`.
- `NnLineage` — Content of `[metadata.nn.lineage]`.
- `NnLoraBranch` — Content of `[metadata.nn.candle.lora]`.
- `NnModelCard` — Aggregate tying a [`CardId`] to its [`NnCardMeta`].
- `NnMoeBranch` — Content of `[metadata.nn.candle.custom.moe]` — the projection of
- `TrainingPath` — Which training path produced a Card, plus the path-specific

## Constants

- `SUPPORTED_ARCHITECTURE_FAMILIES` — Architecture family prefixes accepted for [`NnCardMeta::architecture`].
- `SUPPORTED_TRAINING_PATHS` — Training paths accepted for [`NnCardMeta::training_path`].

