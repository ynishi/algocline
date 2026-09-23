# algocline-nn::merged

Merged inference checkpoint export (Layer 4a of GH #10).

After a LoRA-wrapped model has been trained (via
[`crate::train::run_lora_ft`]), the on-disk delta bundle plus the
base model can be composed into a single safetensors bundle that
downstream consumers load as if it were a plain pretrained model.
This module ships the export half only — a caller-supplied
[`MergedProvenance`] plus a wrapped model produce a safetensors
file on disk (via [`export_merged`]) and a matching
[`NnCardMeta`] describing the merged bundle's provenance.

# Design pattern (§Q0)

[`MergedProvenance`] is the Model-side SoT for what the Card
metadata layer will record. The `to_card_meta` projection maps
into existing [`NnCardMeta`] fields only — the Card schema
itself (`crate::card`) is not modified beyond adding `"merged"`
to the accepted `training_path` values (Layer 4a S4). Future
branches (LoRA / distillation / full-FT hyperparams) can adopt
the same pattern — Model-side struct + `to_card_*` projection —
so the "which Card field do we invent?" negotiation stays local
to each branch's Model-side struct.

See the Layer 4 merged-checkpoint design notes §3 Q0 for the
full pattern rationale.

## Functions

- `export_merged` — Export a merged inference-ready safetensors bundle for a

## Types

- `MergeError` — Errors surfaced by [`export_merged`].
- `MergedProvenance` — Provenance for a merged inference bundle. This is the Model-side

