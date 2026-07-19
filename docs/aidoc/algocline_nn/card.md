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

## Types

- `NnCandleBranch` — Content of `[metadata.nn.candle]`.
- `NnCardMeta` — Content of `[metadata.nn]`.
- `NnLineage` — Content of `[metadata.nn.lineage]`.
- `NnLoraBranch` — Content of `[metadata.nn.candle.lora]`.

