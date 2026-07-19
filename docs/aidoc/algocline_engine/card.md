# algocline-engine::card

Card storage — immutable run-result snapshots.

A Card is a frozen record of a strategy run: identity, parameters,
model, scenario, aggregate stats, and (optionally) per-case detail.
Cards are **immutable** — once written they are never modified, only
annotated via additive `append`.  Mutable **aliases** point to a
Card and can be rebound freely.

## Design principles

1. **Minimal REQUIRED, maximal OPTIONAL** — v0 needs only 4 fields;
   lightweight "ran this pkg" records and heavy optimize snapshots
   share the same schema.
2. **Immutable append-only** — no overwrite, no delete.  New data is
   added via `append` (new top-level keys only) or by creating a new
   Card with a fresh `card_id`.
3. **Two-tier storage** — TOML for human-readable aggregate, JSONL
   sidecar for machine-parseable per-case detail.
4. **File-primary** — files are the source of truth; in-memory state
   is cache.  Cards can be copied, diffed, and version-controlled.

## Storage layout (two-tier)

| Tier | File | Content |
|------|------|---------|
| **Tier 1** | `~/.algocline/cards/{pkg}/{card_id}.toml` | Aggregate scalars, decisions, identity, params |
| **Tier 2** | `~/.algocline/cards/{pkg}/{card_id}.samples.jsonl` | Per-case raw data (JSONL, write-once) |

Tier 1 holds a shareable summary (a few KB). Tier 2 holds per-case
detail ��� the engine does not interpret its columns; packages define
their own schema.

Alias table: `~/.algocline/cards/_aliases.toml` (global).

## card_id naming

`{pkg}_{model_short}_{compact_ts}_{hash6}`

- `compact_ts`: `YYYYMMDDTHHMMSS` in UTC
- `hash6`: first 6 hex chars of DJB2 param fingerprint
- Example: `cot_opus46_20260412T061500_a3f9c1`

## v0 schema (frozen)

### REQUIRED (minimum valid Card)

| Field | Type | Example |
|-------|------|---------|
| `schema_version` | string | `"card/v0"` |
| `card_id` | string | `"cot_opus46_20260412T061500_a3f9c1"` |
| `created_at` | string (RFC 3339) | `"2026-04-12T06:15:00Z"` |
| `[pkg].name` | string | `"cot"` |

### OPTIONAL (auto-injected where possible)

| Section | Fields |
|---------|--------|
| `[pkg]` | `version`, `category`, `source`, `source_ref`, `source_sha` |
| `[runtime]` | `alc_version`, `lua_version`, `host_os`, `git_sha` |
| `[model]` | `provider`, `id`, `id_short`, `cutoff` |
| `[params]` | Free-form ctx snapshot; `param_fingerprint` for DJB2 hash |
| `[strategy_params]` | Strategy-tunable parameters surfaced for sweeps / optimizers (e.g. `alpha`, `temperature`, `depth`). Free-form, but `where`-queryable as a first-class section |
| `[scenario]` | `name`, `source`, `case_count`, `grader` |
| `[stats]` | `pass_rate`, `mean_score`, `std`, `median`, `min`, `max`, `n` |
| `[stats.by_bucket]` | Disaggregated sub-bucket stats (array of tables) |
| `[cost]` | `llm_calls`, `input_tokens`, `output_tokens`, `elapsed_ms`, `usd_estimate` |
| `[optimize]` | `target`, `search`, `rounds_used`, `top_k` (for optimize Cards) |
| `[metadata]` | Free-form escape hatch. Recognized lineage conventions: `prior_card_id` (parent Card id), `prior_relation` (relation kind, e.g. `"sweep_variant"`, `"reflection_of"`, `"derived_from"`) |

## Lua API (`alc.card.*`)

| Function | Description |
|----------|-------------|
| `create(table)` | Write new Card (Tier 1). Returns `{ card_id, path }` |
| `get(card_id)` | Read Card by id. Returns table or nil |
| `list(filter?)` | List Cards as summaries (newest first) |
| `find(query?)` | Prisma-style `where` DSL + dotted-path `order_by` + `offset`/`limit` |
| `append(card_id, fields)` | Additive-only annotation (new keys only) |
| `alias_set(name, card_id, opts?)` | Pin mutable alias |
| `alias_list(filter?)` | List aliases |
| `get_by_alias(name)` | Resolve alias → full Card |
| `write_samples(card_id, samples)` | Write Tier 2 sidecar (write-once) |
| `read_samples(card_id, opts?)` | Read Tier 2 with `where` filtering + offset/limit paging |
| `lineage(query)` | Walk ancestry/descendants via `metadata.prior_card_id` |

## Functions

- `alias_list_with_store` — List aliases from `store`, optionally filtered by pkg.
- `alias_set_with_store` — Bind (or rebind) an alias to a Card in `store`.
- `aliases_to_json` — (no documentation)
- `append_with_store` — Append new top-level fields to an existing Card.
- `card_sink_backfill_with_store` — Backfill one subscriber (`sink` URI) from the primary store.
- `create_with_store` — Create a new Card backed by `store`.
- `eval_predicate` — Evaluate a predicate tree against a full Card JSON.
- `event_bus` — Return the process-wide `CardEventBus` singleton, initializing it
- `find_with_store` — Filter/sort Cards across the store using the `where` DSL.
- `get_by_alias_with_store` — Resolve an alias name to its bound Card and return the full Card JSON.
- `get_with_store` — Read a Card from `store` by id. Returns None if not found.
- `import_from_dir_with_store` — Import Card files into `store` from `source_dir` under `pkg`.
- `init_event_bus` — Eagerly initialize the bus. Idempotent and safe to call multiple
- `lineage_to_json` — Render a LineageResult as JSON for the service layer.
- `lineage_with_store` — Walk the lineage tree from `q.card_id` in `store`.
- `list_with_store` — List cards from `store`. `pkg_filter = Some("name")` restricts to that pkg subdir.
- `parse_order_by` — Parse an order_by JSON value.  Accepts:
- `parse_where` — Parse a `where` JSON value into a `Predicate`.
- `publish` — Convenience wrapper: publish through the singleton.
- `read_samples_with_store` — Read per-case samples from `{card_id}.samples.jsonl`.
- `subscriber_stats_snapshot` — Public entry point: snapshot of all process-wide subscriber stats.
- `summaries_to_json` — (no documentation)
- `validate_name` — (no documentation)
- `write_samples_with_store` — Write per-case samples to `{card_id}.samples.jsonl` (write-once).

## Types

- `Alias` — (no documentation)
- `CardEvent` — A Card-level event emitted from the write path.
- `CardEventBus` — Process-wide fan-out bus. Subscribers are registered once at startup
- `CardEventKind` — Lightweight discriminant for `CardEvent`. Used as a `HashMap` key in
- `CmpOp` — Single comparison operator.
- `Comparison` — One parsed comparison: `path` points at a nested field,
- `FileCardStore` — File-backed implementation of [`CardStore`].
- `FileCardSubscriber` — A subscriber that mirrors events to a local directory using the
- `FindQuery` — Query parameters for `find`.
- `LastError` — Most recent delivery failure for a single subscriber. Exposed via
- `LineageDirection` — Walk direction for `lineage`.
- `LineageEdge` — One edge in the lineage result (child → parent, always).
- `LineageNode` — One node in the lineage result.
- `LineageQuery` — Query parameters for `lineage`.
- `LineageResult` — Full lineage walk result.
- `OrderKey` — Parsed sort key: path with optional descending flag.
- `PerSubscriber` — Per-subscriber counter state. Held inside `SubscriberStats` under a
- `Predicate` — Parsed predicate tree.
- `RunSection` — Optional `[run]` section carrying strategy execution outcome.
- `RunStatus` — Status of a strategy run, recorded in the `[run]` section of a Card.
- `SamplesQuery` — Query parameters for `read_samples`.
- `SinkBackfillReport` — Result of a [`card_sink_backfill`] run. One row per card the tool
- `SubscriberHealthRow` — Snapshot row for a single subscriber, serialized directly into the
- `SubscriberStats` — Process-wide per-subscriber statistics, keyed by subscriber URI
- `Summary` — Summary row for `alc.card.list()`.

## Traits

- `CardStore` — Storage backend for Cards.
- `CardSubscriber` — A downstream backend that receives `CardEvent`s in best-effort,

## Constants

- `SCHEMA_VERSION` — (no documentation)

