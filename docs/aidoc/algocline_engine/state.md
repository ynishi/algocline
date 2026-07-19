# algocline-engine::state

Persistent key-value state backed by JSON files.

## Architecture

All state operations go through the [`StateStore`] trait, which
abstracts the storage backend.  The default implementation,
[`JsonFileStore`], persists each namespace as a JSON file under a
caller-provided root directory with atomic writes (tmp + rename).

## Tier 1 — Current API

| Operation | Description |
|-----------|-------------|
| `get` | Read a value (returns `None` if absent) |
| `set` | Write a value (upsert) |
| `delete` | Remove a key (returns whether it existed) |
| `keys` | List all keys in a namespace |
| `has` | Check existence (cost is backend-dependent) |
| `set_nx` | Set-if-not-exists (returns `false` if key already present) |
| `incr` | Counter increment — single-process atomic (read-modify-write) |

## Tier 2 — Future Extensions (design notes, not yet implemented)

The following operations are planned but **not yet implemented**.
The trait is designed to accommodate them without breaking changes.
Review this list when adding a new backend.

- **TTL**: `set(key, value, opts)` with `opts.ttl_secs`, plus
  `ttl(key) -> Option<Duration>` to query remaining time.  Useful
  for caching patterns (e.g. Hub index cache, LLM response cache).
- **Batch**: `mget(keys) -> Vec<Option<Value>>` and
  `mset(pairs) -> Result<()>`.  Reduces I/O round-trips for
  file/network backends.
- **clear**: Flush all keys in a namespace.  OpenResty's
  `flush_all` equivalent.

## Backend Swappability

Because the engine interacts with state only through the
[`StateStore`] trait, backends can be swapped without changing Lua
code.  Planned backends:

- `JsonFileStore` (current, default)
- In-memory `HashMap` (for tests and short-lived sessions)
- SQLite (for larger datasets with indexed queries)
- Redis (for distributed / multi-process scenarios)

## Types

- `JsonFileStore` — JSON-file-backed state store.
- `ResetReport` — Report returned by [`JsonFileStore::reset_dispatched_with_backup`]
- `StateError` — Errors returned by the dispatched-layout helpers

## Traits

- `StateStore` — Backend-agnostic key-value state store.

