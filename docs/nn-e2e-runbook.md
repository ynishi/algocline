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

Script: `examples/nn_fullft_shakespeare_e2e.lua`.

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

## Cleanup

The script writes only under `cache_dir` (default `target/nn-e2e`),
which is inside `target/` and therefore always gitignored. Remove it
manually if you want a fresh run:

```
rm -rf target/nn-e2e
```
