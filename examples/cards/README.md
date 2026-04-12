# Card examples

Runnable Lua examples that exercise the `alc.card.*` API end-to-end.
Each example is self-contained and can be run via `alc_run` with no
LLM calls required — synthetic scores stand in for real grader output
so the Card flow itself (create / get / list / append / find /
alias_set / alias_list) can be inspected in isolation.

## prompt_ab_demo.lua

Generic LLM-world pattern: A/B-test a matrix of prompt variants
crossed with temperatures against a fixed scenario, record each trial
as an immutable Card, query for the best, pin it with an alias, and
annotate it post-hoc via `append`.

```sh
alc_run code_file=examples/cards/prompt_ab_demo.lua
```

Expected output (shape):

```json
{
  "trials": [
    {"card_id": "prompt_ab_demo_opus46_...", "variant": "terse", "temperature": 0.0, "score": 0.62},
    ...
  ],
  "best": {
    "card_id": "prompt_ab_demo_opus46_...",
    "pass_rate": 0.81
  },
  "aliases": [
    {"name": "best_prompt_ab", "card_id": "...", "pkg": "prompt_ab_demo", "set_at": "..."}
  ],
  "total_cards_for_pkg": 6
}
```

Inspect the resulting files:

```sh
ls ~/.algocline/cards/prompt_ab_demo/
cat ~/.algocline/cards/_aliases.toml
```

## sweep_replay_demo.lua

Replay sweep pattern: load existing Cards (seed data), derive new
Cards with varying EMA alpha parameters, and chain them via
`metadata.prior_card_id`. Demonstrates the two-tier content policy:

- **Tier 1** (Card body): `stats.ev`, `stats.ev_raw`, `params.alpha`
- **Tier 2** (samples.jsonl): per-case `{ case, passed, score }` rows

Creates 3 seed Cards with per-case samples, then derives 12 Cards
(3 seeds × 4 alpha values). Pins the best-scoring Card with an alias.

```sh
alc_run code_file=examples/cards/sweep_replay_demo.lua
```

Inspect:

```sh
ls ~/.algocline/cards/sweep_replay_demo/
cat ~/.algocline/cards/_aliases.toml
```

### Adapting to real workloads

Replace `synthetic_score(...)` with a real grader — either call
`evalframe` directly, or precompute scores from an external eval run
and feed them in. The rest of the Card flow stays identical.

For a real-world integration example see
[`conglo/packages/portfolio_store/init.lua`](https://github.com/yutakanishimura/conglo)
(`task/card-integration` branch), which emits Cards from every
`record_evaluation` call in a biz_kernel pipeline.
