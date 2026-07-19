# algocline-engine::card_context

Card context injection — resolves Card summaries for prompt injection.

Phase 3-F MVP: `alc.llm(prompt, {card_context = ...})` opts key wires
a small set of prior Cards into the LLM system prompt as an XML-like
`<past_cards>` block.  This module is the pure-Rust core: Lua bridge
wiring lives in `bridge/llm.rs`.

## Public API

* [`CardContextSpec`] — 2-form spec (`CardId` or `Query { pkg, limit }`)
  received from the Lua bridge after opts extraction.
* [`resolve`] — spec → `Vec<Json>` full Card resolution against a
  `CardStore`.  Silent Ok(empty) on not-found / empty query.
* [`format_past_cards`] — infallible fixed-template renderer producing
  the `<past_cards>...</past_cards>` block.

## Fixed format template (MVP)

```text
<past_cards>
Card MM/DD pkg=<pkg> card_id=<id> [run.status=<status>] Rating <val> reason=<reason>
...
</past_cards>
```

* Each Card on 1 line, `created_at` desc order (newest first)
* `[run.status=...]` emitted only when the Card JSON carries `run.status`
* `Rating <val>` emitted only when `stats.pass_rate` is present
  (formatted with 1 decimal digit)
* `reason=<reason>` emitted only when `run.reason` is present
* `MM/DD` derived from RFC 3339 `created_at` (`&s[5..10]` with the
  `-` separator rewritten to `/`); omitted when `created_at` is
  missing or too short
* Empty input slice → empty string (no `<past_cards>` wrapper)

## Error handling

`resolve` returns `Result<Vec<Json>, String>`.  Underlying
`card::get_with_store` / `card::find_with_store` errors are re-wrapped
with a `"card_context resolve: "` prefix so the bridge layer can
surface them via `tracing::warn!` before silently dropping the inject
(Phase 3-F Done Criteria #4: silent no-op on resolution failure).

## Functions

- `format_past_cards` — Render a slice of resolved Card JSON values as the fixed
- `resolve` — Resolve a [`CardContextSpec`] against `store`, returning full Card

## Types

- `CardContextSpec` — Card context specification extracted from the Lua `card_context` opt.

