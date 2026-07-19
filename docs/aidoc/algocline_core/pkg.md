# algocline-core::pkg

Canonical projection of a Lua package's `M.meta` block.

`PkgEntity` captures the identity portion of an algocline package: the
fields users rely on to discover, categorize, and version-track a package.
It is the single source of truth for "what is this package?" and is
flattened into higher-level records (`IndexEntry`, `SearchResult`,
`hub_info` responses) so the JSON wire shape stays consistent across the
Hub, the manifest, and project lockfiles.

## Parsing contract

[`PkgEntity::parse_from_init_lua`] is a non-Lua-VM best-effort parser over
the `M.meta = { ... }` block of an `init.lua`. It deliberately only
supports flat key–value pairs with (possibly concatenated) string
literals; nested tables (e.g. `tags = { ... }`) are skipped via
brace-depth tracking. When `M.meta.name` is absent or empty the parser
returns `None` — this is the **inclusion gate** for hub indexing. The
caller (`build_index` in `algocline-app::service::hub`) is expected to
drop `None` directories silently so "draft" directories like
`alc_shapes/` (a type DSL library, not an algocline package) do not
pollute the hub index.

## Wire format

`Option` fields use `#[serde(default)]` but deliberately do **not** use
`skip_serializing_if`. A missing field deserializes as `None` and
serializes back as `null`. This preserves the key-presence guarantee of
the current `hub_index.json` consumers (Bundled-side doc generation,
`README.md` package-count scripts) so they do not break on field
absence.

## Types

- `PkgEntity` — Canonical projection of a Lua package's `M.meta` block.
- `PkgType` — Package type discriminator: runnable (has `M.run`) or library (API surface only).
- `TypeSource` — Records how `pkg_type` was determined for a package.

