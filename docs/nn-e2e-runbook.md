# alc.nn End-to-End Runbook

This runbook drives the public MCP surface of `alc.nn.*` end-to-end
against a real corpus (Tiny Shakespeare) and verifies 15 observable
properties in one alc_run pass.

Unlike the in-harness smokes under `crates/algocline-*/tests/` (which
run inside Rust integration tests with a synthetic 4-row corpus and a
Rust-side Lua VM), this pass exercises the same surfaces a downstream
Lua author would touch: `alc.nn.preset.*`, `alc.nn.data.*`,
`alc.nn.trainer.*`, `alc.nn.card.*`, `alc.nn.sampler.*`,
`alc.card.get`, plus the trainable handles' `generate_session` /
stateless backend.

Scripts:

- `examples/nn_fullft_shakespeare_e2e.lua` — Full FT path (15 phases)
- `examples/nn_lora_shakespeare_e2e.lua` — LoRA path (15 phases, see
  [LoRA path](#lora-path-nn_lora_shakespeare_e2elua) below)
- `examples/nn_distill_llm_teacher_e2e.lua` — Distillation path with a
  live LLM teacher (13 phases, see
  [Distillation path](#distillation-path-nn_distill_llm_teacher_e2elua)
  below)

## When to run it

- After changing anything in `crates/algocline-engine/src/bridge/nn_*`
- After changing anything in `crates/algocline-nn/src/train/` or
  `crates/algocline-nn/src/arch/`
- After a refactor that consolidates or moves the trainer opts
  extractors / error-conversion helpers (the loud-error prefixes in
  Phase 15 catch prefix drift)
- Before a nn-facing release, alongside `just test-nn`

## Prerequisites

- `alc` built and installed with the `nn` feature:
  ```
  cargo install --path . --features nn
  ```
  and the MCP server restarted so the new binary is loaded.
- `curl` on `$PATH` (used to fetch the corpus on first run).
- Approximately 5 MB free under `target/nn-e2e/` for the cached
  corpus and the safetensors checkpoints.

## How to run

The script is designed to be driven via `mcp__algocline__alc_run` (an
MCP client, Claude Code included). Point it at the script file and
pass the algocline repo root as `project_root`:

```json
{
  "code_file": "examples/nn_fullft_shakespeare_e2e.lua",
  "project_root": "<absolute path to the algocline repo>"
}
```

Optional context (JSON object as `ctx`):

- `cache_dir` — override the cache location
  (default: `target/nn-e2e`).
- `corpus_url` — override the Tiny Shakespeare URL.

## What each phase verifies

| # | Phase | Verifies |
|---|---|---|
| 1 | `corpus_fetch` | Corpus downloadable / re-cachable via `io.popen("curl …")` |
| 2 | `tokenize_and_chunk` | 62-codepoint char vocab fits `vocab=64`; 10 000 rows chunked at ctx=16 |
| 3 | `preset_build` | `alc.nn.preset.gpt2("tiny", { pretrained=false })` returns a Handle with the expected shape |
| 4 | `dataset_build` | `alc.nn.data.synthetic` accepts the 10 000-row corpus |
| 5 | `train_completed` | `alc.nn.trainer.full_ft` completes 500 optimizer steps |
| 6 | `loss_descended` | Final loss < 3.2 (baseline `ln(vocab)` = ln 64 ≈ 4.158). Threshold accounts for shuffle-induced ±0.1 variance |
| 7 | `ckpt_file_exists` | Safetensors land on disk under `cache_dir/ckpts/` |
| 8 | `card_saved` | `alc.nn.trainer.run_full_ft` trains **and** registers a Card in one call, returning a non-empty `card_id` string |
| 9 | `card_reload` | `alc.nn.card.load_handle(card_id)` round-trips a usable Handle |
| 10 | `reloaded_generates` | Reloaded Handle exposes `generate_session`; produces 8 tokens through the stateless backend |
| 11 | `greedy_determinism` | Two independent greedy sessions with the same prompt produce byte-identical tokens (no hidden randomness) |
| 12 | `sampler_param_reflected` | `alc.nn.sampler.temperature(1.5, seed=12345)` diverges from greedy within 8 tokens — the sampler parameter is actually reflected in output |
| 13 | `top_k_1_equals_greedy` | `alc.nn.sampler.top_k_top_p(1, …)` is byte-identical to greedy over 12 tokens — sampler contract holds |
| 14 | `train_lr_reflected` | Two independent `run_full_ft` runs with `lr = 1e-5` vs `lr = 3e-3` produce final losses that differ by at least 0.05 — the training hyperparameter is actually reflected in the learning trajectory |
| 15 | `strict_validation_prefixes` | `run_full_ft` refuses (a) missing `lr` / (b) unknown `schedule = "linear"` / (c) negative `warmup`, and every refusal carries the `alc.nn.trainer.run_full_ft` prefix (one prefix per surface — the loud-error contract) |

## Expected output

A successful run returns:

```
{
  "verdict":     "COMPLETED_GREEN",
  "green_count": 15,
  "red_count":   0,
  "card_id":     "shakespeare-tiny-v1_<timestamp>",
  "cache_dir":   "target/nn-e2e",
  "checks":      [ … 15 entries, all with ok=true … ]
}
```

Reference numbers from a CPU (M-series) run:

- Phase 5 wall-clock: ~3 s for 500 steps
- Phase 6 final loss: ~2.9 (best ~2.86)
- Phase 14 loss delta: ~1.5 (`lr = 1e-5` around 4.1, `lr = 3e-3`
  around 2.6)
- Total elapsed: ~5–6 s

## Interpreting failures

- **Phase 6 red (`loss_descended`)** — final loss ≥ 3.2. Either the
  optimizer step regressed (rare) or shuffle variance breached the
  threshold on this seed. Re-run once; if it repeats, inspect
  `crates/algocline-nn/src/train/loop.rs`.
- **Phase 10 red (`reloaded_generates`, "attempt to call a nil
  value")** — the running binary predates the change that exposed
  `generate_session` on trainable handles (RF1). Rebuild with
  `cargo install --path . --features nn` and restart the MCP server.
- **Phase 12 red (`greedy == temperature`)** — sampler parameter did
  not affect the output. Either the sampler registration is broken
  (see `crates/algocline-engine/src/bridge/nn_sampler.rs`) or the
  distribution collapsed to a single token (regression of Phase 6).
- **Phase 14 red (`delta < 0.05`)** — learning rate had no measurable
  effect on the final loss. Investigate the optimizer step application
  path.
- **Phase 15 red (any prefix missing)** — a `run_full_ft` refusal did
  not carry the `alc.nn.trainer.run_full_ft` prefix. The loud-error
  contract regressed — check whether an opts-extraction refactor
  dropped the surface prefix argument.

## LoRA path (`nn_lora_shakespeare_e2e.lua`)

Same corpus, same invocation shape — point `code_file` at
`examples/nn_lora_shakespeare_e2e.lua` instead. Exercises the LoRA
one-call trainer, the Δ-only checkpoint contract, the
`load_wrap` composition path, and the merge export:

| # | Phase | Verifies |
|---|---|---|
| 1-3 | `corpus_fetch` / `tokenize_and_chunk` / `dataset_build` | Same as Full FT phases 1/2/4 |
| 4 | `base_preset_build` | Fresh from-scratch base; dedicated to Phase 5 (run_lora_ft wraps the base model in place, so the handle is never reused) |
| 5 | `lora_train_completed` | `alc.nn.trainer.run_lora_ft` trains the Δ **and** registers a Card in one call (rank=4, alpha=8, default target_modules) |
| 6 | `lora_card_meta_shape` | `training_path="lora"`, `candle.lora` carries rank / alpha / `base_bundle_ref="nn/gpt2-tiny"` / the 6-entry gpt2 default target set, `bundle_ref="nn/<card_id>"` |
| 7 | `lora_loss_descended` | Final loss < 4.05 (baseline ln 64 ≈ 4.158). LoRA has less capacity than Full FT here — the embedding and LM head stay frozen at random init, so the descent is smaller |
| 8 | `delta_file_exists` | The Δ-only safetensors exists at `candle.lora.delta_path`; a LoRA card has **no** `<nn_dir>/<card_id>.safetensors` |
| 9 | `load_handle_refusal` | `load_handle(lora_card_id)` is a designed refusal naming `training_path="lora"` and directing to `load_wrap` |
| 10 | `load_wrap_roundtrip` | `alc.nn.card.load_wrap(card_id, fresh_base)` returns a wrapped NnHandle |
| 11 | `wrapped_generates` | The wrapped handle exposes `generate_session` (stateless backend) |
| 12 | `merge_export` | `alc.nn.card.merge_lora(wrapped, {name, lora_card})` registers a `training_path="merged"` Card with `lineage.parent = lora_card_id` |
| 13 | `merged_load_generates` | The merged Card loads standalone via `load_handle` (no base needed) and generates |
| 14 | `merged_parity_with_wrapped` | 12 greedy tokens byte-identical between the wrapped (base + Δ composed at forward time) and merged (Δ materialised) handles |
| 15 | `strict_validation_prefixes` | `run_lora_ft` refuses (a) missing `rank` / (b) unknown target module / (c) `dropout = 1.5` / (d) an already-wrapped base, each with the `alc.nn.trainer.run_lora_ft` prefix |

Reference numbers from a CPU (M-series) run:

- Phase 7 final loss: ~3.97 (stable to 3 decimals across runs)
- Total elapsed: ~5-7 s

LoRA-specific failure notes:

- **Phase 7 red** — the LoRA Δ did not descend. Check the
  `run_lora_ft` dispatch in
  `crates/algocline-engine/src/bridge/nn_trainer.rs` and the wrap
  path in `crates/algocline-nn/src/arch/lora.rs`.
- **Phase 14 red** — the merged export lost or distorted the Δ.
  Compare with the Rust parity tests
  (`crates/algocline-nn/tests/merged_export_parity_gpt2.rs`).
- **opts gotcha** — the `run_*` trainer family wants `batch` and the
  `"CosineWithWarmup"` / `"Constant"` schedule vocabulary; the
  low-level `alc.nn.trainer.lora` wants `batch_size` and lower-case
  `"cosine"` / `"constant"`. Copy-pasting opts across the two
  families fails on `opts.batch` / `opts.schedule`.

## Distillation path (`nn_distill_llm_teacher_e2e.lua`)

Same invocation shape, but the run **pauses once**: Phase 1 issues a
single `alc.llm_batch` carrying 12 teacher prompts and `alc_run`
returns `status: "needs_response"` with the query batch. Answer the
queries — the sanctioned pattern is dispatching the `@alc-eval` agent
with the desired teacher model (e.g. `model=haiku`) so the teacher is
whichever LLM answers the pause, with no direct API billing — and the
script resumes through Phase 13 with no further pauses. This
exercises the AnyModel-routing shape end to end: the teacher signal
is plain text, which is exactly what the hard-label CE distill loss
consumes (KL soft targets are deferred).

| # | Phase | Verifies |
|---|---|---|
| 1 | `teacher_collect` | One `alc.llm_batch` pause returns a non-empty response per prompt |
| 2 | `teacher_card_created` | `alc.card.create` with Tier 1 `metadata.loss_mask = "response"` |
| 3 | `samples_written` | `alc.card.write_samples` (write-once) lands 48 rows (12 pairs × 4 — `TeacherCardDataset` does not wrap, so rows ≥ steps × batch) |
| 4 | `samples_roundtrip` | `alc.card.read_samples` returns the rows with `prompt` / `response` fields |
| 5 | `dataset_built` | `alc.nn.data.from_card` builds the mask-carrying `TeacherCardDataset` with the real gpt2 tokenizer |
| 6 | `student_built` | `alc.nn.preset.gpt2("custom", { vocab = 50257, … })` — the only small from-scratch config that holds real gpt2 token ids (tiny is vocab=64; medium is 355M) |
| 7 | `distill_completed` | `alc.nn.trainer.run_distill` trains **and** registers a Card in one call |
| 8 | `card_meta_shape` | `training_path="distillation"`, `hyperparams.loss_kind="ce"`, `bundle_ref="nn/<card_id>"` |
| 9 | `loss_descended` | Final loss < 10.5 (baseline ln 50257 ≈ 10.825) |
| 10 | `ckpt_file_exists` | Full student weights at `<nn_dir>/<card_id>.safetensors` (~13 MB) |
| 11 | `custom_load_handle_roundtrip` | `load_handle` rebuilds the custom config from `metadata.nn.candle.custom` and reloads the trained weights standalone (shape accessors + one greedy token verified) |
| 12 | `mask_boundary_guard` | `from_card` loudly refuses a row whose prompt exhausts `ctx_len` (FullyMaskedRow protection) |
| 13 | `strict_validation_prefixes` | `run_distill` refuses missing `lr` / `loss_kind = "kl"` / `schedule = "linear"`, each with the `alc.nn.trainer.run_distill` prefix |

Reference numbers from CPU (M-series) runs with a Haiku teacher:

- Phase 9 final loss: ~7.1-7.6 (teacher responses vary per run)
- Post-resume elapsed: ~30-75 s (dominated by the 40 distill steps
  over the vocab-50257 embedding)

Distill-specific notes:

- First run fetches `gpt2.json` from the HF hub into
  `<nn_dir>/tokenizers/`; later runs are fully offline.
- `run_distill` shares the strict `run_*` opts family: `batch` (not
  `batch_size`), `"CosineWithWarmup"` / `"Constant"` schedule
  vocabulary, `loss_kind` accepts only `"ce"`.
- Custom cards record their shape under `metadata.nn.candle.custom`,
  so `load_handle` reloads them standalone (Phase 11). Cards trained
  by a build that predates this metadata are still refused with a
  directional error; custom+MoE cards are refused until an MoE
  reload path lands.

## Cleanup

The Shakespeare scripts write only under `cache_dir` (default `target/nn-e2e`),
which is inside `target/` and therefore always gitignored. Remove it
manually if you want a fresh run:

```
rm -rf target/nn-e2e
```

The one-call trainers additionally register Cards (and write
safetensors) under `~/.algocline/nn/` on every run; prune old
`shakespeare-*` cards there if the accumulation bothers you.
